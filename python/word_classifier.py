import json


class WordClassifier():
    def __init__(self, dict_files_dir):
        self.dict_files_dir_ = dict_files_dir


    def readDictFile(self, file_path):
        with open(file_path) as f:
            content = json.load(f)
            print(content)










