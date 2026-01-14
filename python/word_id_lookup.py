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
        # print(f"self.word_to_id_file_path_: {self.word_to_id_file_path_}")

        atexit.register(self.saveWordIdFile)

        if not self.word_to_id_file_path_.is_file():
            self.saveWordIdFile()

        self._initReservedIds()
        self._buildLookup()

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


