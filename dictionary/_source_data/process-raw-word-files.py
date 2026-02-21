import os.path
import json
import copy
import pathlib

dictionary_path     = pathlib.Path(__file__).parent.parent.resolve()
source_data_path    = (dictionary_path / "_source_data").resolve()

source_names = [
    "nouns_irregular", "nouns_regular", "02_nouns",   # Nouns first. Rest in alphabetical order
    "adjectives", "conjunctive_adverbs", "degree_adverbs", "frequency_adverbs", "manner_adverbs",
    "place_adverbs", "time_adverbs", "verbs_irregular", "verbs_regular",
    "auxiliary_verbs", "misc_words",
    "02_verbs"
]

print(f"dictionary_path : {dictionary_path}")
print(f"source_data_path: {source_data_path}")
print(f"source_names    : {source_names}")

global g_total_words
g_total_words  =0


def merge_dict_into_dict(word_into, word_from):
    if word_into['base'] != word_from['base']:
        print(f"ERROR: {word_into['base']} != {word_from['base']}")
        return False

    for key in word_from:
        if key == "classes":
            for class_name in word_from['classes']:
                if not class_name in word_into['classes']:
                    word_into['classes'].append(class_name)
        else:
            if word_from[key] != "":
                word_into[key] = word_from[key]

    if 'n' in word_into['classes']:
        word_into['classes'].remove('n')
        word_into['classes'].insert(0, "n")

    if 'n0' in word_into:
        word_into.pop('n0', None)

    return True


def word_file_path(base_name):
    first_letter = base_name[0]
    return dictionary_path / first_letter / f"{base_name}._.json"

def read_word_file(base_name, wfp):
    if not os.path.exists(wfp):
        pathlib.Path(wfp).parent.mkdir(parents=True, exist_ok=True)
        return { "base": base_name, "classes": [] }

    with open(wfp) as f:
        return json.load(f)


def merge_dict_into_file(word_from):
    global g_total_words
    g_total_words += 1
    base_name = word_from['base']
    wfp = word_file_path(base_name)
    word_into = read_word_file(base_name, wfp)
    merge_dict_into_dict(word_into, word_from)
    print(f"WORD INTO [{g_total_words}]: {word_into} -> {wfp}")
    with open(wfp, "w") as fp:
        json.dump(word_into, fp, indent=4)


def merge_source_file(source_name):
    source_file_path = source_data_path / f"{source_name}.json"
    if not os.path.exists(source_file_path):
        print(f"ERROR: {source_file_path} does not exist")
        return

    with open(source_file_path) as f:
        words_from = json.load(f)
        for words_from in words_from:
            merge_dict_into_file(words_from)



for source_name in source_names:
    print(f"Processing file source_name: {source_name}")
    merge_source_file(source_name)

print (f"\n--- Total words updated: {g_total_words} ---\n")

# wi1 = {
#     "base": "dark",
#     "classes": ["n"],
#     "n1": "dark",
#     "n2": "darks",
#     "n3": "the dark",
#     "n4": "the darks",
#     "n5": "dark's"
#
# }
#
# wf1 = {
#     "base": "dark",
#     "classes": ["adj"],
#     "adj1": "dark",
#     "adj2": "darker",
#     "adj3": "darkest"
# }
#
# print(f"BEFORE wi1: {wi1}\n")
#
# w1 = merge_dict_into_dict(wi1, wf1)
#
# print(f"AFTER  wi1: {wi1}\n")


