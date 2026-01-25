import random

from sympy.strategies.core import switch


def initFiles(trainFilePath, validateFilePath):
    with open(trainFilePath, "w") as f:
        f.write("")
    with open(validateFilePath, "w") as f:
        f.write("")

def startPrompt():
    return " <prompt> "

def endPrompt():
    s = ""
    if random.randint(1,3) == 3:
        s += " ?"
    s += " </prompt> <response> "
    return s

def endResponse():
    return " . </response> "


def traingen_math_plus(a, b, choice: int = None):
    NUM_VARIANTS = 5
    choiceQ = choice
    choiceA = choice

    if choiceQ is None:
        choiceQ = random.randint(1, NUM_VARIANTS)
    res = a + b
    s = startPrompt()

    match choiceQ:
        case 1: s += f"What is {a} + {b}"
        case 2: s += f"Please calculate the following: {a} + {b}"
        case 3: s += f"How much is {a} + {b}"
        case 4: s += f"What is the total of {a} + {b}"
        case 5: s += f"What is the answer to {a} + {b}"

    s += endPrompt()

    if choiceA is None:
        choiceA = random.randint(1, NUM_VARIANTS)
    match choiceA:
        case 1: s += f"The result is {res}"
        case 2: s += f"Correct answer is {res}"
        case 3: s += f"{res} is the result"
        case 4: s += f"The answer is {res}"
        case 5: s += f"The sum is {res}"

    s += endResponse()

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

    s = startPrompt()

    match choiceQ:
        case 1: s += f"What is {a} - {b}"
        case 2: s += f"Please calculate the following: {a} - {b}"
        case 3: s += f"How much is {a} - {b}"
        case 4: s += f"What is the difference of {a} - {b}"
        case 5: s += f"What is the answer to {a} - {b}"

    s += endPrompt()

    if choiceA is None:
        choiceA = random.randint(1, NUM_VARIANTS)
    match choiceA:
        case 1: s += f"The result is {res}"
        case 2: s += f"Correct answer is {res}"
        case 3: s += f"{res} is the result"
        case 4: s += f"The answer is {res}"
        case 5: s += f"The difference is {res}"

    s += endResponse()

    return s


def traingen_math_multiply(a, b, choice: int = None):
    NUM_VARIANTS = 5
    choiceQ = choice
    choiceA = choice

    if choiceQ is None:
        choiceQ = random.randint(1, NUM_VARIANTS)
    res = a * b
    s = startPrompt()

    match choiceQ:
        case 1: s += f"What is {a} * {b}"
        case 2: s += f"Please calculate the following: {a} * {b}"
        case 3: s += f"How much is {a} * {b}"
        case 4: s += f"What is the multiplication of {a} * {b}"
        case 5: s += f"What is the answer to {a} * {b}"

    s += endPrompt()

    NUM_VARIANTS = 4
    if choiceA is None:
        choiceA = random.randint(1, NUM_VARIANTS)
    match choiceA:
        case 1: s += f"The result is {res}"
        case 2: s += f"Correct answer is {res}"
        case 3: s += f"{res} is the result"
        case 4: s += f"The answer is {res}"

    s += endResponse()

    return s


# Note: To (for now) keep this as integers, we muliply the two numbers given and create a division
# question by dividing the result with one of the factors
def traingen_math_divide(a, b, choice: int = None):
    NUM_VARIANTS = 5
    choiceQ = choice
    choiceA = choice

    if choiceQ is None:
        choiceQ = random.randint(1, NUM_VARIANTS)
    nominator = a * b
    s = startPrompt()

    match choiceQ:
        case 1: s += f"What is {nominator} / {b}"
        case 2: s += f"Please calculate the following: {nominator} / {b}"
        case 3: s += f"How much is {nominator} / {b}"
        case 4: s += f"What is the multiplication of {nominator} / {b}"
        case 5: s += f"What is the answer to {nominator} / {b}"

    s += endPrompt()

    NUM_VARIANTS = 4
    if choiceA is None:
        choiceA = random.randint(1, NUM_VARIANTS)
    match choiceA:
        case 1: s += f"The result is {a}"
        case 2: s += f"Correct answer is {a}"
        case 3: s += f"{a} is the result"
        case 4: s += f"The answer is {a}"

    s += endResponse()

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

    s = startPrompt()

    match choiceQ:
        case 1: s += f"What number comes after {a}, {b}"
        case 2: s += f"Next integer following {a}, {b} is"
        case 3: s += f"Given the sequence {a}, {b}. What is the expected next number"
        case 4: s += f"If we have {a} followed by {b}, the next number would be"
        case 5: s += f"What number naturally comes after {a}"
        case 6: s += f"Next integer following {a} is"

    s += endPrompt()

    NUM_VARIANTS = 5
    if choiceA is None:
        choiceA = random.randint(1, NUM_VARIANTS)
    match choiceA:
        case 1: s += f"The next number is {res}"
        case 2: s += f"Correct answer is {res}"
        case 3: s += f"{res} is the number"
        case 4: s += f"The answer is {res}"
        case 5: s += f"The next is {res}"

    s += endResponse()

    return s


def traingen_math_before_1(a, choice: int = None):
    NUM_VARIANTS = 3
    choiceQ = choice
    choiceA = choice

    if choiceQ is None:
        choiceQ = random.randint(1, NUM_VARIANTS)
    # b = a + 1
    res = a - 1
    s = startPrompt()

    match choiceQ:
        case 1: s += f"What number comes before {a}"
        case 2: s += f"Integer that comes before {a} is"
        case 3: s += f"What number naturally comes before {a}"

    s += endPrompt()

    NUM_VARIANTS = 4
    if choiceA is None:
        choiceA = random.randint(1, NUM_VARIANTS)
    match choiceA:
        case 1: s += f"The prior number is {res}"
        case 2: s += f"Correct answer is {res}"
        case 3: s += f"{res} is the number"
        case 4: s += f"The answer is {res}"

    s += endResponse()

    return s