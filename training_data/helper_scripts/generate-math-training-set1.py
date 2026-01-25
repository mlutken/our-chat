import random
from traingen_math import *
import pathlib

training_data_path     = pathlib.Path(__file__).parent.parent.resolve()
print (f"training_data_path: {training_data_path}")

trainFilePath = training_data_path / "math-validation-set1.txt"
validateFilePath = training_data_path / "math-training-set1.txt"


MIN_RANDOM_NUMBER = 0
MAX_RANDOM_NUMBER = 100
PLUS_SENTENCES_COUNT = 500

initFiles(trainFilePath, validateFilePath)

for n in range(1, PLUS_SENTENCES_COUNT):
    with open(trainFilePath, "a+") as ft, open(validateFilePath, "a+") as fv:
        s = ""
        a = random.randint(MIN_RANDOM_NUMBER, MAX_RANDOM_NUMBER)
        b = random.randint(MIN_RANDOM_NUMBER, MAX_RANDOM_NUMBER)
        s += traingen_math_plus(a, b)
        s += "\n"
        # Split out first 10% for validation
        if n <= PLUS_SENTENCES_COUNT/10:
            fv.write(s)
        else:
            ft.write(s)

