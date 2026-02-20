import sys
import glob
import os
import json
import torch

# from tensorflow.python.ops.inplace_ops import empty

from system_globals import *
from word_id_lookup import *
from tokenizer_utils import *

# from nltk.misc import word_finder


class DictionaryTokenizer():
    def __init__(self, dict_files_dir):
        self.dict_files_dir_ = os.path.realpath(dict_files_dir)
        self.templates_files_dir_ = os.path.join(self.dict_files_dir_, "_templates")
        self.templates_input_files_dir_ = os.path.join(self.dict_files_dir_, "_templates_input")
        word_ids_file_path = os.path.join(self.dict_files_dir_, "word_ids.json")

        self.wordIdLookup_ = WordIdLookup(word_ids_file_path)

        self.enum_ids_path_ = self.dict_files_dir_ + "/enum_ids.json"
        # self.debugPrintInfo()
        self._buildEnumIds()
        self._buildTokenizer()

        # Number bits default settings
        self.number_bits_integer_part = 0
        self.number_bits_fractional_part = 0
        self.min_binary_number = 0
        self.max_binary_number = 0
        self.setNumberBitsIntPart(16)    # To also set min and max

        self.max_token_id_ = self.wordIdLookup_.maxWordId()
        self.max_token_value = self.max_token_id_   # Only temporarely needed !!

    def setConfig(self, config):
        self.config_ = config
        self.setNumberBitsIntPart(self.config_["number_bits"])

    def setNumberBitsIntPart(self, number_bits):
        self.number_bits_integer_part = number_bits
        #self.number_bits_fractional_part = number_bits # TODO: Support fixed point numbers
        self.min_binary_number = minBinaryNumber(self.number_bits_integer_part)
        self.max_binary_number = maxBinaryNumber(self.number_bits_integer_part)

    def debugPrintInfo(self):
        print(f"self.dict_files_dir_          : {self.dict_files_dir_}")
        print(f"self.enum_ids_path_           : {self.enum_ids_path_}")

    def vocabSize(self):
        return self.maxTokenId() + 1

    def maxTokenId(self):
        return self.max_token_id_

    def baseWordToId(self, baseWord):
        return self.wordIdLookup_.baseWordToId(baseWord)

    def idToBaseWord(self, baseWordId):
        return self.wordIdLookup_.idToBaseWord(baseWordId)

    def isKnownWord(self, word):
        subtree = self.tokenizerTree_
        for ch in word:
            if ch not in subtree:
                return False
            subtree = subtree[ch]
            if not subtree:
                return False

        return True

    # def idsToWord(self, baseWordId, wordClassId):
    #     return self.wordIdLookup_.idToBaseWord(id)

    def processIdBatchForEmbedding(self, idsInBatch):
        # print(f"FIXMENM processIdBatchForEmbedding idsInBatch.dtype: {idsInBatch.dtype}")
        idsOut = []
        numberIndices = []
        batch_count = int(idsInBatch.size(0))
        tokens_size = int(idsInBatch.size(1))

        batchNumber = 0

        while batchNumber < batch_count:
            processedIds = self._processIdVectorForEmbedding(idsInBatch[batchNumber])
            numberIndices.append(processedIds["number_indices"])
            idsOut.append(processedIds["ids"])
            batchNumber += 1


        ids_tensor = torch.zeros(batch_count, tokens_size, dtype=torch.float64)
        for batch_number in range(batch_count):
            token_index = 0
            tokens_in_batch = len(idsOut[batch_number])
            while token_index < tokens_in_batch:
                token_id = idsOut[batch_number][token_index]
                try:
                    ids_tensor[batch_number, token_index] = token_id
                except Exception as error:
                    print(f"FIXMENM batch_number: {batch_number}, token_index: {token_index}, token_id: {token_id}")
                    print("An error occurred:", error)  # An error occurred: name 'x' is not defined
                    ids_tensor[batch_number, token_index] = PADDING_ID
                    raise error

                # print(f"FIXMENM batch_number: {batch_number}, token_index: {token_index}, token_id: {token_id}, {ids_tensor[batch_number, token_index]}")
                token_index += 1


        bin_numbers = torch.zeros(batch_count, tokens_size, self.number_bits_integer_part)
        for batch_number in range(batch_count):
            numIndices = numberIndices[batch_number]
            for number_index in numIndices:
                decimal_number = idsOut[batch_number][number_index]
                ids_tensor[batch_number, number_index] = int(VALUE_ID)
                binary_number = decimalToBinaryTensor(decimal_number, self.number_bits_integer_part, self.number_bits_fractional_part)
                bin_numbers[batch_number, number_index,:] = binary_number
                # print(f"FIXMENM --> token_id: {token_id}, number_index: {number_index}, decimal: {decimal_number} == {decimal_number2}, binary: {binary_number}")
                # print(f"FIXMENM --> token_id: {token_id}, number_index: {number_index}, decimal: {decimal_number}, binary: {binary_number}")
                # print(f"FIXMENM --> bin_numbers[batch_number, number_index]: {bin_numbers[batch_number, number_index]}")
                # dec_numbers_FIXMENM[batch_number, token_index+1] = decimal_number
                # if decimal_number == 42:
                #     print(f"FIXMENM ------------- token_id: {token_id}, token_index: {token_index}, decimal: {decimal_number}, binary: {binary_number}")
                # print(f"FIXMENM batch_number: {batch_number}, {number_index} => {ids_tensor[batch_number, number_index]}")

        ids_tensor = ids_tensor.to(dtype=torch.int32)
        return {'ids': ids_tensor, "number_indices": numberIndices, 'bin_numbers': bin_numbers, 'idsOut': idsOut}   # idsOut is currently for convenience/debugging

    ## allowed_special is not currently used. Only because the tiktoken tokenizer has this
    def encode(self, text, allowed_special={}):
        ids = self.parseTextToIds(text)
        return ids

    def decode(self,  ids):
        return self.idsToText(ids)

    def parseTextToIds(self, text):
        self.addAnyUnknownWords(text)
        words = self.parseTextToWords(text)
        return self.wordsToIds(words)

    def idsToText(self, ids):
        wordList = self.idsToWordList(ids)
        return self.wordListToText(wordList)

    def parseTextToWords(self, text):
        text = text.lower()
        words = []
        currentTxtPos = 0
        txtLen = len(text)
        while currentTxtPos < txtLen:
            # print (f"FIXMENM currentTxtPos: {currentTxtPos}")
            word, currentTxtPos = self.parseNextWord(text, currentTxtPos)
            words.append(word)
        return words

    def parseNextWord(self, text, currentTxtPos):
        txtLen = len(text)
        if currentTxtPos >= txtLen:
            return '', currentTxtPos

        word, currentTxtPos = self.tryParseNextWordAsNumber(text, currentTxtPos)
        if word == "":
            word, currentTxtPos = self.parseNextWordFromParseTree(text, currentTxtPos)

        return word, currentTxtPos

    def addAnyUnknownWords(self, text):
        word = ''
        for ch in text:
            ch = ch.lower()
            if isSeparator(ch):
                if (not self.isKnownWord(word)) and (not self._tryParseNumber(word)):
                    self.wordIdLookup_.checkAddToUnknownWords(word)
                word = ''
            else:
                word += ch


    def tryParseNextWordAsNumber(self, text, currentTxtPos):
        txtLen = len(text)
        saveTextPos = currentTxtPos
        numberStr = ""
        while (currentTxtPos < txtLen) and ( isNumberChar(text[currentTxtPos]) ):
            numberStr = numberStr + text[currentTxtPos]
            currentTxtPos = currentTxtPos + 1

        if numberStr == "":
            return "", saveTextPos

        if not self._canParseAsNumber(numberStr):
            return "", saveTextPos

        return numberStr, currentTxtPos


    def parseNextWordFromParseTree(self, text, currentTxtPos):
        txtLen = len(text)
        if currentTxtPos >= txtLen:
            return '', currentTxtPos
        word = ""

        parseStack = []
        parseStack.append({'treeLvl' : self.tokenizerTree_, 'txtPos': currentTxtPos})

        # print(f"FIXMENM parseStack[0]: {parseStack[0]}")

        self.tokenizerTreeParseAsFarAsPossible(parseStack, text, txtLen)
        parseStackEntry = parseStack[-1]
        while parseStack and (not treeEntryHasWord(parseStackEntry['treeLvl'])):
            parseStack.pop(-1)
            if parseStack:
                parseStackEntry = parseStack[-1]

        # print(f"FIXMENM AFTER parseStack len: {len(parseStack)}")
        # print(f"FIXMENM parseStackEntry: {parseStackEntry}")

        if 'word' in parseStackEntry['treeLvl']:
            word = parseStackEntry['treeLvl']['word']
            currentTxtPos = parseStackEntry['txtPos']
        else:
            word = "<u>"
            currentTxtPos += 1

        return word, currentTxtPos

    def tokenizerTreeParseAsFarAsPossible(self, parseStack, text, textLen):
        while self.treeLevelHasNextChar(parseStack[-1], text, textLen):
            txtPos = parseStack[-1]['txtPos']
            ch = text[txtPos]
            # print(f"FIXMENM [{ch}] parseStack len: {len(parseStack)}, txtPos: {txtPos}")
            treLvl = parseStack[-1]['treeLvl']
            treLvlNext = treLvl[ch]
            parseStack.append({'treeLvl': treLvlNext, 'txtPos': txtPos + 1})

    def treeLevelHasNextChar(self, parseStackEntry, text, textLen):
        txtPos = parseStackEntry['txtPos']
        if txtPos >= textLen:
            return False

        treeLvl = parseStackEntry['treeLvl']
        if text[txtPos] in treeLvl:
            return True

        return False

    def wordToIds(self, word):
        wordAsNumber = self._tryParseNumber(word)
        if wordAsNumber is not None:
            return [NUMBER_ID, wordAsNumber]

        subtree = self._lookupWord(word)
        if not subtree:
            return []

        return subtree['ids']

    def wordsToIds(self, words):
        ids = []
        for word in words:
            wordIds = self.wordToIds(word)
            for wordId in wordIds:
                ids.append(wordId)
        return ids

    def wordListToText(self, wordList):
        text = ""
        for word in wordList:
            text = text + word
        return text

    ''' For succesfull conversion: 
        First ID MUST be a baseWord or other start special ID '''
    def idsToWordList(self, ids):
        currentPos = 0
        idsCount = len(ids)
        words = []
        while currentPos < idsCount:
            nextWord, currentPos = self.nextWordFromIds(ids, currentPos)
            words.append(nextWord)
        return words

    ''' For succesfull conversion: 
        First ID MUST be a baseWord or other start special ID '''
    def nextWordFromIds(self, ids, currentPos):
        idsCount = len(ids)
        if currentPos >= idsCount:
            return '', currentPos

        # --- Look for next base word ID ---
        while currentPos < idsCount and not self.wordIdLookup_.idIsBaseWord(ids[currentPos]):
            currentPos = currentPos + 1

        if currentPos >= idsCount:
            return '', currentPos

        baseWordId = ids[currentPos]

        # Check for number
        if idIsNumber(baseWordId):
            currentPos = currentPos + 1
            if currentPos >= idsCount:
                return '', currentPos
            word = str(ids[currentPos])
            currentPos = currentPos + 1
        else:
            word = self.wordIdLookup_.idToBaseWord(baseWordId)

            currentPos = currentPos + 1
            if currentPos >= idsCount:
                return word, currentPos

            classFormId = ids[currentPos]
            if baseWordId in self.idsToWordlookup_:
                if classFormId in self.idsToWordlookup_[baseWordId]:
                    word = self.idsToWordlookup_[baseWordId][classFormId]

        while currentPos < idsCount and not self.wordIdLookup_.idIsBaseWord(ids[currentPos]):
            currentPos = currentPos + 1

        return word, currentPos

    # def tokenize(self, str):
    #     tokens = []
    #     return tokens

    def listAllFiles(self):
        print("listAllFiles")
        for path in glob.glob(self.dict_files_dir_ + "/**/*._.json", recursive = True):
        #for path in glob.glob("/home/ml/code/crawler/machine-learning/dictionary/**/*.json", recursive = True):
            print(f"path: {path}")

    def saveTokenizerTree(self, fileName):
        with open(fileName, "w") as fp:
            json.dump(self.tokenizerTree_, fp, indent=4)

    def saveIdsToWordlookup(self, fileName):
        with open(fileName, "w") as fp:
            json.dump(self.idsToWordlookup_, fp, indent=4)

    def getClassIdsFromClassNames(self, classes):
        class_ids = []
        for class_name in classes:
            class_ids.append(self.classNameToId_[class_name])
        return class_ids

    def getFormIdFromForms(self, forms):
        return self.formToId_[forms[0]]

    def dbgTestParseTextToWords(self, text, verbosePrint=False):
        words = self.parseTextToWords(text)
        # print (f"FIXMENM words: {words}")
        ids = self.wordsToIds(words)
        textRecreated = self.idsToText(ids)
        if verbosePrint:
            print (f"PARSE      : \"{text}\" => {words} =>\n                 {ids} =>\nIDS TO TEXT: \"{textRecreated}\"")
        else:
            print (f"PARSE: \"{text}\" => \"{textRecreated}\"")


    def dbgTestWordsToText(self, words):
        ids = self.wordsToIds(words)
        textBefore = self.wordListToText(words)
        textAfter = self.idsToText(ids)
        print (f"DBG: dbgTestWordsToText {textBefore} => {textAfter}")

    def dbgTestSingleWord(self, word):
        ids = self.wordToIds(word)
        wordFromIds, currentPos = self.nextWordFromIds(ids, 0)
        print (f"DBG: dbgTestSingleWord [{currentPos}]: {word} -> {ids} => {wordFromIds}")


    def dbgTestSingleBaseWord(self, word):
        idFromWord = self.baseWordToId(word)
        wordFromId = self.idToBaseWord(idFromWord)
        errStr = ''
        if wordFromId != word:
            errStr = f" ; ERROR: ID -> word '{wordFromId}'"
        print(f"'{word}' -> '{idFromWord}'{errStr}")

    def classFormToId(self, classForm):
        return self.formToId_[classForm]

    def idToClassForm(self, classFormId):
        return self.idToForm_[classFormId]

    def _tryParseNumber(self, numberStr):
        try:
            number = float(numberStr)
            if (number < self.min_binary_number) or (self.max_binary_number < number ):
                return None
            return number
        except:
            return None

    def _canParseAsNumber(self, numberStr):
        number = self._tryParseNumber(numberStr)
        if number is None:
            return False
        return True

    def _lookupWord(self, word):
        subtree = self.tokenizerTree_
        for ch in word:
            subtree = subtree[ch]
            if not subtree:
                return {}

        return subtree

    def _buildTokenizerFromTemplate(self, templateName):
        pathTemplate = self._getTemplatePath(templateName)
        pathInput = self._getTemplateInputPath(templateName)
        # print(f"FIXMENM {templateName} :{pathTemplate}, {pathInput}")
        template = None
        inputData = None
        with open(pathTemplate) as json_file:
            template = json.load(json_file)
        with open(pathInput) as json_file:
            inputData = json.load(json_file)

        for word in inputData["words"]:
            dictEntry = template
            dictEntry["base"] = word
            dictEntry["lc"] = word
            # print(f"FIXMENM dictEntry:\n{dictEntry}")
            self._buildTokenizerOneDictEntry(dictEntry)

        # print(f"FIXMENM template:\n{template}")
        # print(f"FIXMENM inputData:\n{inputData}")

    def _buildTokenizer(self):
        self._buildTokenizerFromTemplate("LETTER")
        self._buildTokenizerFromTemplate("CHARACTER")

        self._buildTokenizerAddSpecialWord(PADDING_TOKEN, "pad", "f0")
        self._buildTokenizerAddSpecialWord(START_TOKEN, "sep", "f0")
        self._buildTokenizerAddSpecialWord(END_TOKEN, "sep", "f0")
        self._buildTokenizerAddSpecialWord(UNKNOWN_TOKEN, "unk", "f0")
        self._buildTokenizerAddSpecialWord(PROMPT_START_TOKEN, "sep", "f0")
        self._buildTokenizerAddSpecialWord(PROMPT_END_TOKEN, "sep", "f0")
        self._buildTokenizerAddSpecialWord(RESPONSE_START_TOKEN, "sep", "f0")
        self._buildTokenizerAddSpecialWord(RESPONSE_END_TOKEN, "sep", "f0")
        self._buildTokenizerAddSpecialWord(CALC_EVAL_START_TOKEN, "sep", "f0")
        self._buildTokenizerAddSpecialWord(CALC_EVAL_END_TOKEN, "sep", "f0")
        self._buildTokenizerAddSpecialWord(CONTEXT_START_TOKEN, "sep", "f0")
        self._buildTokenizerAddSpecialWord(CONTEXT_END_TOKEN, "sep", "f0")

        for path in glob.glob(self.dict_files_dir_ + "/**/*._.json", recursive = True):
            should_open_file = "/_templates" not in path
            # print(f"FIXMENM should_open_file [{should_open_file}]: {path}")
            if should_open_file:
                with open(path) as json_file:
                    dictEntry = json.load(json_file)
                    self._buildTokenizerOneDictEntry(dictEntry)

    def _buildTokenizerOneDictEntry(self, dictEntry):
        # print(f"------ FIXMENM dict: {dictEntry} ------")
        self._buildIdsToWordLookup(dictEntry)
        self.wordIdLookup_.addWord(dictEntry["base"])

        for classType in dictEntry["classes"]:
            # print(f"FIXMENM classType: '{classType}'")
            if classType in self.CLASS_TO_FORM_NAMES_:
                for form in self.CLASS_TO_FORM_NAMES_[classType]:
                    if form in dictEntry:
                        word = dictEntry[form]
                        # print(f"FIXMENM form: '{form}'")
                        # print(f"FIXMENM word: '{word}'")
                        self._buildTokenizerAddCharsInWord(word, classType, form, dictEntry["base"])

    def _buildTokenizerAddSpecialWord(self, word, classType, wForm):
        wBase = word
        self._buildTokenizerAddCharsInWord(word, classType, wForm, wBase)

    def _buildTokenizerAddCharsInWord(self, word, classType, wForm, wBase):
        # print(f"FIXMENM word: '{word}' t: {classType}, f: {wForm}")
        if word == "":
            return

        currentTreeLvl = self.tokenizerTree_
        for c in word:
            if not c in currentTreeLvl:
                currentTreeLvl[c] = {}
            currentTreeLvl = currentTreeLvl[c]

        currentTreeLvl["word"] = word
        currentTreeLvl["base"] = wBase

        if not "classforms" in currentTreeLvl:           currentTreeLvl["classforms"] = [wForm]
        elif not wForm in currentTreeLvl["classforms"]:  currentTreeLvl["classforms"].append(wForm)

        form_id = self.getFormIdFromForms(currentTreeLvl["classforms"])
        currentTreeLvl["ids"] = [self.baseWordToId(wBase)]
        if not (wForm == "f0"):
            currentTreeLvl["ids"].append(form_id)


        if self.buildTokenizerDoAddClasses_:
            if not "classes" in currentTreeLvl:                 currentTreeLvl["classes"] = [classType]
            elif not classType in currentTreeLvl["classes"]:    currentTreeLvl["classes"].append(classType)

            class_ids = self.getClassIdsFromClassNames(currentTreeLvl["classes"])
            for class_id in class_ids:
                currentTreeLvl["ids"].append(class_id)


    def _buildIdsToWordLookup(self, dictEntry):
        baseWord = dictEntry["base"]
        baseWordId = self.wordIdLookup_.baseWordToId(baseWord)
        self.idsToWordlookup_[baseWordId] = {
            "base": baseWord
        }

        for classType in dictEntry["classes"]:
            # print(f"FIXMENM classType: '{classType}'")
            if classType in self.CLASS_TO_FORM_NAMES_:
                for classForm in self.CLASS_TO_FORM_NAMES_[classType]:
                    if classForm in dictEntry:
                        wordForm = dictEntry[classForm]
                        formId = 0
                        formId = self.classFormToId(classForm)
                        # print(f"FIXMENM [{classForm}]: '{wordForm}' -> {formId}")
                        self.idsToWordlookup_[baseWordId][formId] = wordForm

    def _buildEnumIds(self):
        with open(self.enum_ids_path_) as json_file:
            dict = json.load(json_file)
            self.classNameToId_ = dict["classes"]
            self.formToId_ = dict["classforms"]

            for form_name, form_id in self.formToId_.items():
                self.idToForm_[form_id] = form_name

    def _getTemplatePath(self, templateName):
        return os.path.join(self.templates_files_dir_, f"{templateName}._.json")

    def _getTemplateInputPath(self, templateName):
        return os.path.join(self.templates_input_files_dir_, f"{templateName}.json")


    # @staticmethod
    def _processIdVectorForEmbedding(self, idsIn):
        idsOut = []
        numberIndices = []
        N = len(idsIn)
        i = 0

        while i < N:
            if idIsNumber(idsIn[i]):
                if i < N - 1:
                    idsOut.append(idsIn[i].item())  # Append NUMBER_ID token
                    i = i + 1   # SKIP NUMBER_ID token
                    idsOut.append(idsIn[i].item())
                    idOutIndex = len(idsOut) - 1
                    numberIndices.append(idOutIndex)
                    i = i + 1
                else:
                    idsOut.append(idsIn[i].item())
                    i = i + 1
            else:
                idsOut.append(idsIn[i].item())
                i = i + 1

        return {'ids': idsOut, "number_indices": numberIndices}


    buildTokenizerDoAddClasses_ = False
    tokenizerTree_ = {}
    idsToWordlookup_ = {}
    classNameToId_ = {}
    formToId_ = {}
    idToForm_ = {}


    CLASS_TO_FORM_NAMES_ = {
        "n": ["n1", "n2", "n3", "n4", "n5"],
        "v": ["v1", "v2", "v3", "v4", "v5"],
        "adj": ["adj1", "adj2", "adj3"],
        "adv": ["f0"],
        "pre": ["f0"],
        "pro": ["f0"],
        "det": ["f0"],
        "conj": ["f0"],
        "intj": ["f0"],
        "l": ["lc", "uc"],
        "i": [],
        "f": [],
        "sep": ["f0"],
        "unk": ["f0"],
        "pad": ["f0"]
    }
