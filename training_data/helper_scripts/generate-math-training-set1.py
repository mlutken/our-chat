import random
from traingen_math import *
import pathlib

training_data_path     = pathlib.Path(__file__).parent.parent.resolve()
print (f"training_data_path: {training_data_path}")

MIN_RANDOM_NUMBER = 0
MAX_RANDOM_NUMBER = 100
PLUS_SENTENCES_COUNT = 500
s = ""

for n in range(1, PLUS_SENTENCES_COUNT):
    a = random.randint(MIN_RANDOM_NUMBER, MAX_RANDOM_NUMBER)
    b = random.randint(MIN_RANDOM_NUMBER, MAX_RANDOM_NUMBER)
    s += traingen_math_plus(a, b)
    if n % 3 == 0:
        s += "\n"


# print(s)

with open( training_data_path / "math-training-set1.txt", "w") as f:
    f.write(s)
