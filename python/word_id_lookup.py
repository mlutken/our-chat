import sys
import glob
import os
import json
from pathlib import Path
import atexit
from tokenizer_utils import *

# Solution - About Python Unable to use Open functions in the destructor
# https://www.programmersought.com/article/41489941223/


class WordIdLookup():
    START_RESERVED_IDS_     = START_RESERVED_IDS
    START_AUTO_ID_          = START_AUTO_ID

    def __init__(self, word_to_id_file_path):
        self.word_to_id_file_path_ = Path(word_to_id_file_path)
        self.unknown_words_file_path_ = Path("/tmp/_unknown_words.json")
        self.unknown_words_sorted_file_path_ = Path("/tmp/_unknown_words_sorted_counts.txt")
        self.unknown_words_sorted_text_file_path_ = Path("/tmp/_unknown_words_sorted.txt")
        # print(f"self.word_to_id_file_path_: {self.word_to_id_file_path_}")
        self.unknown_words_ = {}

        atexit.register(self.saveWordIdFile)

        if not self.word_to_id_file_path_.is_file():
            self.saveWordIdFile()

        self._initReservedIds()
        self._buildLookup()

    def checkAddToUnknownWords(self, word):
        if (word == "") or (self.isKnownBaseWord(word)):
            return

        # print(f"FIXMENM B checkAddToUnknownWords: '{word}'")
        if word in self.unknown_words_:
            self.unknown_words_[word] += 1
        else:
            self.unknown_words_[word] = 1

    def isKnownBaseWord(self, word):
        return word in self.word_to_id_

    def maxWordId(self):
        return self._lastId

    def baseWordToId(self, baseWord):
        return self.word_to_id_.get(baseWord, 0)

    def idToBaseWord(self, wordId):
        return self.id_to_word_.get(wordId, "")

    def isBaseWord(self, word):
        return self.baseWordToId(word) != 0

    def idIsBaseWord(self, id):
        return id >= self.START_RESERVED_IDS_

    def addWord(self, word):
        if not word in self.word_to_id_:
            self._lastId += 1
            self.word_to_id_[word] = self._lastId
            self.id_to_word_[self._lastId] = word

    def saveWordIdFile(self):
        with open(self.word_to_id_file_path_, "w") as fp:
            json.dump(self.word_to_id_, fp, indent=4)
        with open(self.unknown_words_file_path_, "w") as fp:
            json.dump(self.unknown_words_, fp, indent=4)

        # Sort by number of occurrences and save
        listSorted = []
        for word, cnt in self.unknown_words_.items():
            listSorted.append([word, cnt])

        listSorted.sort(key=lambda x: x[1], reverse=True)
        with open(self.unknown_words_sorted_file_path_, "w") as fp:
            for wordTuple in listSorted:
                word, cnt = wordTuple
                fp.write(f"{word} : {cnt}\n")

        with open(self.unknown_words_sorted_text_file_path_, "w") as fp:
            for wordTuple in listSorted:
                word, cnt = wordTuple
                fp.write(word + "\n")

    def _buildLookup(self):
        self._initWordToIdDict()

        # Build id to word opposite lookup table
        self._lastId = self.START_AUTO_ID_ - 1
        for word, wordId in self.word_to_id_.items():
            if wordId > self._lastId:
                self._lastId = wordId
            self.id_to_word_[wordId] = word


    def _initWordToIdDict(self):
        with open(self.word_to_id_file_path_) as json_file:
            self.word_to_id_ = json.load(json_file)

        for word, id in self.reserved_word_to_id_.items():
            self.word_to_id_[word] = id

    def _initReservedIds(self):
        self.reserved_word_to_id_[PADDING_TOKEN] = PADDING_ID
        self.reserved_word_to_id_[START_TOKEN] = START_TOKEN_ID
        self.reserved_word_to_id_[END_TOKEN] = END_TOKEN_ID
        self.reserved_word_to_id_[UNKNOWN_TOKEN] = UNKNOWN_TOKEN_TOKEN_ID
        self.reserved_word_to_id_[PROMPT_START_TOKEN] = PROMPT_START_ID
        self.reserved_word_to_id_[PROMPT_END_TOKEN] = PROMPT_END_ID
        self.reserved_word_to_id_[RESPONSE_START_TOKEN] = RESPONSE_START_ID
        self.reserved_word_to_id_[RESPONSE_END_TOKEN] = RESPONSE_END_ID
        self.reserved_word_to_id_[CALC_EVAL_START_TOKEN] = CALC_EVAL_START_ID
        self.reserved_word_to_id_[CALC_EVAL_END_TOKEN] = CALC_EVAL_END_ID


    word_to_id_file_path_   = ""
    reserved_word_to_id_ = {}
    word_to_id_ = {}
    id_to_word_ = {}
    unknown_words_ = {}


