import os.path
import json
import copy
import pathlib

g_commoness_minimum_value = 100

dictionary_path     = pathlib.Path(__file__).parent.parent.resolve()
source_data_path    = (dictionary_path / "_source_data").resolve()

source_names = [
    "nouns_irregular", "nouns_regular", "02_nouns", "_wordnet_nouns_forms",   # Nouns first. Rest in alphabetical order
    "verbs_irregular", "verbs_regular", "02_verbs", "_wordnet_verbs_forms"
    "adjectives", "conjunctive_adverbs", "degree_adverbs", "frequency_adverbs", "manner_adverbs", "relative_adverbs", "adverbs",
    "_wordnet_adjective_forms",
    "place_adverbs", "time_adverbs",
    "conjunctions",
    "auxiliary_verbs", "misc_words", "pronoun_words", "preposition_words"
]

print(f"dictionary_path : {dictionary_path}")
print(f"source_data_path: {source_data_path}")
print(f"source_names    : {source_names}")

global g_total_words
g_total_words  =0

def str_contains_number(str):
    return any(char.isdigit() for char in str)

def merge_dict_into_dict(word_into, word_from):
    if word_into['base'] != word_from['base']:
        print(f"ERROR: Failed merging:  {word_into['base']} != {word_from['base']}")
        print(f"into: {word_into}\nfrom: {word_from}\n")

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
    base_name = word_from['base']

    if str_contains_number(base_name):
        print(f"NOTE: skipping word with number (for now at least) '{base_name}'")
        return

    wfp = word_file_path(base_name)
    word_into = read_word_file(base_name, wfp)

    if "commonness" in word_from:
        commonness = int(word_from['commonness'])
        if commonness < g_commoness_minimum_value:
            return
        # else:
        #     print(f"FIXMENM WE WOULD IMPORT '{base_name}' WORD INTO [{g_total_words}]: {word_from} -> {wfp}")
        #     return

    g_total_words += 1
    merge_dict_into_dict(word_into, word_from)
    print(f"WORD INTO [{g_total_words}]: {word_from} -> {wfp}")
    with open(wfp, "w") as fp:
        json.dump(word_into, fp, indent=4)


def merge_source_file(source_name):
    source_file_path_jsonl = source_data_path / f"{source_name}.jsonl"
    source_file_path_json  = source_data_path / f"{source_name}.json"
    if  os.path.exists(source_file_path_json):
        with open(source_file_path_json) as f:
            words_from = json.load(f)
            for words_from in words_from:
                merge_dict_into_file(words_from)
    elif  os.path.exists(source_file_path_jsonl):
        with open(source_file_path_jsonl) as f:
            for line in f:
                try:
                    line = line.strip()
                    if line == "":
                        continue
                    words_from = json.loads(line)
                except:
                    pass
                # print(f"words_from: {words_from}")
                merge_dict_into_file(words_from)
    else:
        print(f"ERROR: {source_file_path_json} or {source_file_path_jsonl} not found!")


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


