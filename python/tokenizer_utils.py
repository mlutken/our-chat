import math
import numpy as np
import torch
from sympy.testing.pytest import ExceptionInfo

NUMBER_ID               = 20
VALUE_ID                = 21

PADDING_ID              = 22
START_TOKEN_ID          = 23
END_TOKEN_ID            = 24
UNKNOWN_TOKEN_TOKEN_ID  = 25
PROMPT_START_ID         = 26
PROMPT_END_ID           = 27
RESPONSE_START_ID       = 28
RESPONSE_END_ID         = 29
CALC_EVAL_START_ID      = 30
CALC_EVAL_END_ID        = 31

START_RESERVED_IDS      =  NUMBER_ID
START_AUTO_ID           = 40

PADDING_TOKEN           = '<pad>'   # '<PADDING>'
START_TOKEN             = '<s>'     # † <s> '<START>'
END_TOKEN               = '</s>'    # ‡ </s> '<END>'
UNKNOWN_TOKEN           = '<u>'     # ‡ <u> '<U>'
PROMPT_START_TOKEN      = '<prompt>'
PROMPT_END_TOKEN        = '</prompt>'
RESPONSE_START_TOKEN    = '<response>'
RESPONSE_END_TOKEN      = '</response>'
CALC_EVAL_START_TOKEN   = '<calc_eval>'
CALC_EVAL_END_TOKEN     = '</calc_eval>'

SEPARATORS_HMM = [
    ' ', "\n", '-', '_',
    '.', ',', '?', '!', ':', ';',
    "'", '"',
    '(', ')', '[', ']', '{', '}'
]

SEPARATORS = [
    ' ', "\n", '.', ',', '?', '!', ':', ';'
]


NUMBER_CHARS = [ '0', '1', '2', '3', '4', '5', '6',  '7', '8', '9', '.', '+', '-']


def idIsNumber(wordId):
    return wordId == NUMBER_ID


def isSpace(ch):
    return ch == ' '


def isSeparator(ch):
    return ch in SEPARATORS

def isNumberChar(ch):
    return ch in NUMBER_CHARS

def maxBinaryNumber(num_int_bits=8):
    return 2**(num_int_bits-1) -1

def minBinaryNumber(num_int_bits=8):
    return -(2**(num_int_bits-1))


def binaryTensorToDecimal(binary_tensor):
    num_int_bits = binary_tensor.shape[-1]
    binary_tensor_slow = binary_tensor.clone().detach()
    if binary_tensor_slow.dim() > 1:
        binary_tensor_slow = binary_tensor[0]

    decimal_slow = 0
    # 2's complement: Add all "positive" bits, which are all bits except most significant bit (MSB)
    for n in range(num_int_bits - 1, 0, -1):
        n_pow = num_int_bits - n - 1
        if binary_tensor_slow[n] >= 0.5:
            decimal_slow += math.pow(2, n_pow)

    # 2's complement: Add MSB with largest negative value
    if binary_tensor_slow[0] >= 0.5:
        decimal_slow -= math.pow(2, num_int_bits-1)
    decimal_slow_tensor = torch.tensor(float(decimal_slow), device=binary_tensor.device).clone().detach().unsqueeze(0).clone().detach()

    # # TODO: This is some rounding errors or similar I have yet to understand why happens.
    # # Does not work when num_bits > 54. Need to figure out what goes wrong with MSB (most significant bit)
    # # THis method should really be faster than the above loop, but it's converting wrongly when num_bits > 54
    # # like for example:
    # # -1 (-> to binary) (-> to decimal) = 0
    # # -255 (-> to binary) (-> to decimal) = -256
    # #
    # num_int_bits = binary_tensor.shape[-1]
    # bits = torch.arange(num_int_bits).to(torch.float64)
    # weights = torch.pow(2, bits).to(torch.float64).flip(dims=[0]).to(binary_tensor.device) # flip reverses elements
    # weights[0] = -weights[0]    # MSB means negative and has -2^num_bits weight in 2s complement
    # bin_tensor = (binary_tensor >= 0.5).to(torch.float64)  # Convert to 0 or 1 en every element
    # decimal = (bin_tensor * weights).sum(-1)    # This fails when num_bits > 54 :(
    # # print (f"FIXMENM CHECK binaryTensorToDecimal decimal != decimal_slow: {decimal} != {decimal_slow}")
    # if decimal != decimal_slow:
    #     print (f"ERROR binaryTensorToDecimal decimal != decimal_slow: {decimal} != {decimal_slow}")
    #     raise ValueError(f"ERROR binaryTensorToDecimal decimal != decimal_slow: {decimal} != {decimal_slow}")

    return decimal_slow_tensor


# See https://stackoverflow.com/questions/1848700/biggest-integer-that-can-be-stored-in-a-double
def decimalToBinaryTensor(decimal, num_int_bits, num_fractional_bits, dtype=torch.float32):
    isNegative = decimal < 0
    fractionalPart, integerPart = math.modf(decimal)
    integerPart = int(integerPart)

    # Initialize a list to hold the bits
    bits = []
    for i in range(num_int_bits - 1, -1, -1):
        # Extract the i-th bit
        bit = (integerPart >> i) & 1
        bits.append(bit)

    # Convert the list to a tensor
    binary_tensor = torch.tensor(bits, dtype=dtype)

    # decimal_check_FIXMENM = binaryTensorToDecimal(binary_tensor)
    # print(f"FIXMENM decimalToBinaryTensor[{num_int_bits}] : [{decimal}  == {decimal_check_FIXMENM}] --> {integerPart} -> {binary_tensor} ")

    return binary_tensor

# def isAsciiLowercaseLetter(ch):
#     return ch in LETTERS


def countAllWordsInFile(file_path):
    dict = {}
    with open(file_path) as f:
        for line in f:
            word = ''
            for ch in line:
                ch = ch.lower()
                if isSeparator(ch):
                    # print(f"Word: '{word}'")

                    if word != "":
                        if not word in dict:
                            dict[word] = 1
                        else:
                            # print(f"FIXMENM word: '{word}'")
                            dict[word] = dict[word] + 1
                        word = ''
                else:
                    word += ch

    return dict


def treeEntryHasWord(treeLvl):
    return 'word' in treeLvl

def saveTensorToFile(filename, tensor):
    np.savetxt(filename, tensor.cpu().numpy(), delimiter=',')

#
# class Entry():
#     def __init__(self, entry_dict):
#         self.entry_dict_ = entry_dict
#
#     def get_form_id(self):


