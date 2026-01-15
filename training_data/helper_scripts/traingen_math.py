import random

from sympy.strategies.core import switch


def traingen_math_plus(a, b, choice: int = None):
    NUM_VARIANTS = 5
    choiceQ = choice
    choiceA = choice

    if choiceQ is None:
        choiceQ = random.randint(1, NUM_VARIANTS)
    res = a + b
    s = ""

    match choiceQ:
        case 1: s += f"What is {a} + {b}"
        case 2: s += f"Please calculate the following: {a} + {b}"
        case 3: s += f"How much is {a} + {b}"
        case 4: s += f"What is the total of {a} + {b}"
        case 5: s += f"What is the answer to {a} + {b}"

    s += "?"
    s += " "

    if choiceA is None:
        choiceA = random.randint(1, NUM_VARIANTS)
    match choiceA:
        case 1: s += f"The result is {res}"
        case 2: s += f"Correct answer is {res}"
        case 3: s += f"{res} is the result"
        case 4: s += f"The answer is {res}"
        case 5: s += f"The sum is {res}"

    s += "!"
    s += " "

    return s


def traingen_math_minus(a, b, choice: int = None):
    NUM_VARIANTS = 5
    choiceQ = choice
    choiceA = choice

    if choiceQ is None:
        choiceQ = random.randint(1, NUM_VARIANTS)
    res = a - b

    # FIXME: We still do not support negative numbers. Should be simple though to add!
    if res < 0:
        a, b = b, a     # SWAP
        res = a - b

    s = ""

    match choiceQ:
        case 1: s += f"What is {a} - {b}"
        case 2: s += f"Please calculate the following: {a} - {b}"
        case 3: s += f"How much is {a} - {b}"
        case 4: s += f"What is the difference of {a} - {b}"
        case 5: s += f"What is the answer to {a} - {b}"

    s += "?"
    s += " "

    if choiceA is None:
        choiceA = random.randint(1, NUM_VARIANTS)
    match choiceA:
        case 1: s += f"The result is {res}"
        case 2: s += f"Correct answer is {res}"
        case 3: s += f"{res} is the result"
        case 4: s += f"The answer is {res}"
        case 5: s += f"The difference is {res}"

    s += "!"
    s += " "

    return s


def traingen_math_multiply(a, b, choice: int = None):
    NUM_VARIANTS = 5
    choiceQ = choice
    choiceA = choice

    if choiceQ is None:
        choiceQ = random.randint(1, NUM_VARIANTS)
    res = a * b
    s = ""

    match choiceQ:
        case 1: s += f"What is {a} * {b}"
        case 2: s += f"Please calculate the following: {a} * {b}"
        case 3: s += f"How much is {a} * {b}"
        case 4: s += f"What is the multiplication of {a} * {b}"
        case 5: s += f"What is the answer to {a} * {b}"

    s += "?"
    s += " "

    NUM_VARIANTS = 4
    if choiceA is None:
        choiceA = random.randint(1, NUM_VARIANTS)
    match choiceA:
        case 1: s += f"The result is {res}"
        case 2: s += f"Correct answer is {res}"
        case 3: s += f"{res} is the result"
        case 4: s += f"The answer is {res}"

    s += "!"
    s += " "

    return s



def traingen_math_next_after_1(a, choice: int = None):
    NUM_VARIANTS = 6
    choiceQ = choice
    choiceA = choice

    if choiceQ is None:
        choiceQ = random.randint(1, NUM_VARIANTS)
    b = a + 1
    res = b + 1
    if choiceQ > 4:
        res = a + 1

    s = ""

    match choiceQ:
        case 1: s += f"What number comes after {a}, {b}"
        case 2: s += f"Next integer following {a}, {b} is"
        case 3: s += f"Given the sequence {a}, {b}. What is the expected next number"
        case 4: s += f"If we have {a} followed by {b}, the next number would be"
        case 5: s += f"What number naturally comes after {a}"
        case 6: s += f"Next integer following {a} is"

    s += "?"
    s += " "

    NUM_VARIANTS = 5
    if choiceA is None:
        choiceA = random.randint(1, NUM_VARIANTS)
    match choiceA:
        case 1: s += f"The next number is {res}"
        case 2: s += f"Correct answer is {res}"
        case 3: s += f"{res} is the number"
        case 4: s += f"The answer is {res}"
        case 5: s += f"The next is {res}"

    s += "!"
    s += " "

    return s


def traingen_math_before_1(a, choice: int = None):
    NUM_VARIANTS = 3
    choiceQ = choice
    choiceA = choice

    if choiceQ is None:
        choiceQ = random.randint(1, NUM_VARIANTS)
    # b = a + 1
    res = a - 1
    s = ""

    match choiceQ:
        case 1: s += f"What number comes before {a}"
        case 2: s += f"Integer that comes before {a} is"
        case 3: s += f"What number naturally comes before {a}"

    s += "?"
    s += " "

    NUM_VARIANTS = 4
    if choiceA is None:
        choiceA = random.randint(1, NUM_VARIANTS)
    match choiceA:
        case 1: s += f"The prior number is {res}"
        case 2: s += f"Correct answer is {res}"
        case 3: s += f"{res} is the number"
        case 4: s += f"The answer is {res}"

    s += "!"
    s += " "

    return s