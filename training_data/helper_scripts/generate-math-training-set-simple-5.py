import random
from traingen_math import *
import pathlib

g_avoid_negative_numbers = False

training_data_path     = pathlib.Path(__file__).parent.parent.resolve()
print (f"training_data_path: {training_data_path}")

trainFilePath = training_data_path / "math-training-simple-5.txt"
validateFilePath = training_data_path / "math-validation-simple-5.txt"

MIN_RANDOM_NUMBER = -1000
MAX_RANDOM_NUMBER = 1000
PLUS_SENTENCES_COUNT = 1000000

initFiles(trainFilePath, validateFilePath)

s = ""

for n in range(1, PLUS_SENTENCES_COUNT):
    with open(trainFilePath, "a+") as ft, open(validateFilePath, "a+") as fv:
        s = ""
        a = random.randint(MIN_RANDOM_NUMBER, MAX_RANDOM_NUMBER)
        b = random.randint(MIN_RANDOM_NUMBER, MAX_RANDOM_NUMBER)
        choice = None
        s += traingen_math_plus(a, b, choice)
        s += "\n"
        s += traingen_math_minus(a, b, choice)
        s += "\n"
        s += traingen_math_multiply(a, b, choice)
        s += "\n"
        s += traingen_math_divide(a, b, choice)
        s += "\n"
        s += traingen_math_next_after_1(a, choice)
        s += "\n"
        s += traingen_math_before_1(a, choice)
        s += "\n"

        # Split out first 10% for validation
        if n <= PLUS_SENTENCES_COUNT / 10:
            fv.write(s)
        else:
            ft.write(s)

