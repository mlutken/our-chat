import torch
import argparse
import os.path
import sys



import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

from nnutils import *
from model_trainer import *
from system_globals import *
from dictionary_tokenizer import *
import pathlib

global g_repo_root_path
global g_python_code_path
g_repo_root_path        = pathlib.Path(__file__).parent.parent.resolve()
g_python_code_path      = pathlib.Path(__file__).parent.resolve()
g_dictionary_path       = g_repo_root_path / "dictionary"
g_training_data_path    = g_repo_root_path / "training_data"

print (f"g_repo_root_path       : {g_repo_root_path}")
print (f"g_python_code_path     : {g_python_code_path}")
print (f"g_dictionary_path      : {g_dictionary_path}")
print (f"g_training_data_path   : {g_training_data_path}")

# Load save model: https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
# time python our-chat.py --epochs 7 --batch_size 12 --load_model 1 --save_model 1 --start_context "<prompt> What is 12 + 4 ? </prompt> " --train_uri ../training_data/math-training-simple-1.txt
# time python our-chat.py --epochs 60 --batch_size 12 --records_to_process -1 --start_context '<prompt> What is 12 + 4 ? </prompt> ' --load_model 1 --save_model 1 --train_uri ../training_data/math-training-simple-4.txt --validation_uri ../training_data/math-validation-simple-4.txt --device cuda --dbg_print_text 0
# time python our-chat.py --epochs 1 --batch_size 12 --records_to_process 140000 --start_context 'I would like to' --load_model 1 --save_model 1 --train_uri 'hf:HuggingFaceFW/finewiki' --dataset_name en --device cuda --dbg_write_records_to_file 0 --dbg_print_text 1
# time python our-chat.py --epochs 1 --batch_size 12 --records_to_process 10000 --start_context 'I would like to' --load_model 1 --save_model 1 --train_uri 'hf:HuggingFaceFW/fineweb' --dataset_name CC-MAIN-2025-26 --device cuda --dbg_write_records_to_file 0 --dbg_print_text 1
# time python our-chat.py --epochs 1 --batch_size 12 --records_to_process 10000 --start_context 'I would like to' --load_model 1 --save_model 1 --train_uri 'hf:Fredithefish/Instruction-Tuning-with-GPT-4-RedPajama-Chat' --dataset_name default --device cuda --dbg_write_records_to_file 0 --dbg_print_text 1

# time python our-chat.py --epochs 60 --batch_size 12 --records_to_process -1 --start_context '<prompt> What is 12 + 4 ? </prompt> ' --load_model 0 --save_model 1 --train_uri ../training_data/math-training-simple-4.txt --device cuda --model_path m2025-12-24.pth --dbg_print_text 1
# time python our-chat.py --epochs 10 --batch_size 12 --records_to_process -1 --start_context '<prompt> What is 12 + 4 ? </prompt> ' --load_model 0 --save_model 1 --train_uri ../training_data/math-training-simple-1.txt --device cuda --model_path m2025-12-24.pth --dbg_print_text 1
# time python our-chat.py --epochs 10 --batch_size 12 --records_to_process -1 --start_context '<prompt> What is 12 + 4 ? </prompt> ' --load_model 1 --save_model 0 --train_uri ../training_data/math-training-simple-2.txt --validation_uri ../training_data/math-validation-simple-2.txt --device cuda --dbg_print_text 0
# g_device = "cpu"

# time python our-chat.py --epochs 2 --batch_size 12 --records_to_process 3 --start_context 'I would like to' --load_model 1 --save_model 0 --train_uri 'hf:HuggingFaceFW/finewiki' --dataset_name en --device cuda --dbg_write_records_to_file 0 --dbg_print_text 1
# time python our-chat.py --epochs 2 --batch_size 12 --records_to_process 3 --start_context 'I would like to' --load_model 1 --save_model 0 --train_uri 'hf:HuggingFaceFW/fineweb' --dataset_name CC-MAIN-2025-26 --device cuda --dbg_write_records_to_file 0 --dbg_print_text 1
# time python our-chat.py --epochs 2 --batch_size 12 --records_to_process 5 --start_context 'I would like to' --load_model 1 --save_model 0 --train_uri ../training_data/math-training-simple-1.txt --dataset_name CC-MAIN-2025-26 --device cuda --dbg_write_records_to_file 0 --dbg_print_text 1
#
# time python our-chat.py --epochs 2 --batch_size 12 --records_to_process 17 --start_context 'I would like to' --load_model 1 --save_model 0 --train_uri 'hf:HuggingFaceFW/fineweb' --dataset_name CC-MAIN-2025-26 --device cuda --dbg_write_records_to_file 0 --dbg_print_text 1 --print_initial_loss 1
# time python our-chat.py --epochs 2 --batch_size 12 --records_to_process -1 --start_context 'I would like to' --load_model 1 --save_model 0 --train_uri ../training_data/math-training-simple-2.txt --dataset_name CC-MAIN-2025-26 --device cuda --dbg_write_records_to_file 0 --dbg_print_text 1 --print_initial_loss 1
# python our-chat.py --run_mode chat-simple


# Crash in PositionalEncoding: 'hf:HuggingFaceFW/finewiki'
# HuggingFace.RECORD[149699] text[0:20]: '# Deadly Women
# Deadl'


# '/home/ml/temp/_FIXMENM_hf_all.txt'

# --- arguments ---
# see: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def print_tested_hugging_face_foundation_training_sets():
    print("*********************************************************")
    print("*** Tested Hugging Face foundation training data sets ***")
    print("*********************************************************")
    print(f"--- 'hf:HuggingFaceFW/finewiki' (https://huggingface.co/datasets/HuggingFaceFW/finewiki) ---")
    print(f"    dataset_name = 'en'; dataset_key = 'text'")
    print("")
    print(f"--- 'hf:HuggingFaceFW/fineweb' (https://huggingface.co/datasets/HuggingFaceFW/fineweb) ---")
    print(f"    dataset_name = 'CC-MAIN-2025-26'; dataset_key = 'text'")
    print("")


def print_usage_examples(exe_name):
    print("**********************")
    print("*** usage examples ***")
    print("**********************")
    print(f"--- Example 1: ---\npython {exe_name} --epochs 10 --batch_size 12 --start_context 'What is 21 + 15?' --load_model 1 --save_model 1 --train_uri ../training_data/math-training-simple-2.txt --validation_uri ../training_data/math-validation-simple-1.txt --device cuda --model_path pass0_dropOutAll2.pth\n")
    print(f"--- Example 2: ---\npython {exe_name} --epochs 10 --batch_size 12 --start_context 'I would like to' --load_model 1 --save_model 1 --train_uri ../training_data/the-verdict.txt  --device cuda --model_path pass0_dropOutAll2.pth\n")
    print(f"--- Example 3: ---\npython {exe_name} --epochs 1 --batch_size 12 --records_to_process 1000 --start_context 'I would like to' --load_model 1 --save_model 1 --train_uri 'hf:HuggingFaceFW/finewiki' --dataset_name en --device cuda --model_path mymodel.pth\n" )
    print(f"--- Example 4: ---\npython {exe_name} --epochs 1 --batch_size 12 --records_to_process 1000 --start_context 'I would like to' --load_model 1 --save_model 1 --train_uri 'hf:HuggingFaceFW/fineweb' --dataset_name CC-MAIN-2024-10 --device cuda --model_path mymodel.pth\n" )

    print(f"--- Example Chat simple: ---\npython {exe_name}  --run_mode chat-simple\n" )

    print("")
    print_tested_hugging_face_foundation_training_sets()


parser = argparse.ArgumentParser("gpt2-simple-train")
parser.add_argument("--device", help="Set device 'cuda' or 'cpu'", nargs='?', type=str, default='')
parser.add_argument("--epochs", help="Number of epochs", nargs='?', type=int, default=0)
parser.add_argument("--plot", help="Plot losses", nargs='?', type=str2bool, const=True, default=False)
parser.add_argument("--records_to_process", help="Maximum number of records to process during training. -1 means all records in training data. Mainly relevant with large streaming ('hf:xx') URIs from HuggingFace", nargs='?', type=int, default=-1)
parser.add_argument("--batch_size", help="Batch size", nargs='?', type=int, default=12)
parser.add_argument("--save_model", help="Save the model after training", nargs='?', type=str2bool, const=True, default=True)
parser.add_argument("--load_model", help="Load model before training", nargs='?', type=str2bool, const=True, default=True)
parser.add_argument("--model_path", help="Model save/load file name", nargs='?', type=str, default="_model.pth")
parser.add_argument("--run_mode", help="Run mode: train, chat-simple", nargs='?', type=str, default="train")
parser.add_argument("--start_context", help="Start context for during training print of generation", nargs='?', type=str, default="<prompt> What is 15 + 5 ? </prompt> ")
parser.add_argument("--train_uri", help="File/URL with training data. Ex.: ../training_data/math-training-simple-2.txt", nargs='?', type=str, default="")
parser.add_argument("--validation_uri", help="File/URL with validation data. Ex.: ../training_data/math-validation-simple-1.txt", nargs='?', type=str, default="")
parser.add_argument("--dataset_name", help="Internal name of Hugging face) dataset: Eg: 'en', 'CC-MAIN-2024-10', ... Depends on concrete dataset.", nargs='?', type=str, default="en")
parser.add_argument("--dataset_key", help="Dictionary key name of primary text data in each record. Eg.: 'text'", nargs='?', type=str, default="text")
parser.add_argument("--dataset_training_split", help="Training split name. Eg.:'train'", nargs='?', type=str, default="train")
parser.add_argument("--dataset_validation_split", help="Training split name. Eg.:'validation'", nargs='?', type=str, default="train")
parser.add_argument("--num_workers", help="Number of worker threads. Not tested", nargs='?', type=int, default=0)
parser.add_argument("--print_initial_loss", help="Calculate and print initial model training and validation losses", nargs='?', type=str2bool, const=True, default=True)
parser.add_argument("--dbg_print_text", help="Debug print streaming (text) records text field. Only first 20 chars", nargs='?', type=str2bool, const=True, default=False)
parser.add_argument("--dbg_write_records_to_file", help="Write streaming (text) records to a file for debug/info. Default file name is '/tmp/_our_streaming_records_debug.txt' ", nargs='?', type=str2bool, const=True, default=False)
parser.add_argument("--dbg_records_file_name", help="Name to write streaming records text to. Default is: '/tmp/_our_streaming_records_debug.txt'", nargs='?', type=str, default="/tmp/_our_streaming_records_debug.txt")
parser.add_argument("--eval_freq", help="Number of batches between each evaluation test of model ", nargs='?', type=int, default=100)
parser.add_argument("--eval_batches", help="Number of batches to run during evaluation", nargs='?', type=int, default=5)
parser.add_argument("--usage", help="Print usage examples", nargs='?', type=str2bool, const=True, default=False)

args = parser.parse_args()
g_dl_data_train = dataloader_lookup(args.train_uri)
g_dl_data_validate = dataloader_lookup(args.validation_uri)




if args.usage:
    print_usage_examples(sys.argv[0])
    sys.exit(0)

device = g_device

if args.device != "":
    device = args.device

num_epochs = args.epochs
## file_path = "../training_data/the-verdict.txt"

default_start_context = args.start_context



print("Default device           : ", g_device)
print("device                   : ", device)
print("run_mode                 : ", args.run_mode)
print("num_epochs               : ", args.epochs)
print("plot                     : ", args.plot)
print("batch_size               : ", args.batch_size)
print("records_to_process       : ", args.records_to_process)
print("save_model               : ", args.save_model)
print("load_model               : ", args.load_model)
print("model_path               : ", args.model_path)
print("default_start_context    : ", default_start_context)
print("train_uri                : ", args.train_uri)
print("validation_uri           : ", args.validation_uri)
print("dataset_name             : ", args.dataset_name)
print("dataset_key              : ", args.dataset_key)
print("dataset_training_split   : ", args.dataset_training_split)
print("dataset_validation_split : ", args.dataset_validation_split)
print("dbg_write_records_to_file: ", args.dbg_write_records_to_file)
print("dbg_records_file_name    : ", args.dbg_records_file_name)
print("num_workers              : ", args.num_workers)
print("print_initial_loss       : ", args.print_initial_loss)
print("dbg_print_text           : ", args.dbg_print_text)
print("eval_freq                : ", args.eval_freq)
print("eval_batches             : ", args.eval_batches)

tokenizer = DictionaryTokenizer("../dictionary")
tokenizer.saveTokenizerTree("/home/ml/temp/_tokenizer_tree.json")
tokenizer.saveIdsToWordlookup("/home/ml/temp/_ids_to_word_lookup.json")
vocab_size = tokenizer.vocabSize()

print (f"FIXMENM tokenizer.max_token_value: {tokenizer.max_token_value}")

emb_dim = 144   # 768, 144, 156
GPT_CONFIG_124M = {
    "vocab_size": vocab_size,
    "context_length": 256,
    "emb_dim": emb_dim, # 768
    "n_heads": 6,   # 12
    "n_layers": 6,  # 12
    "drop_rate": 0.1,
    "qkv_bias": False,
    "number_bits": 32,    #
    "feed_forward_layer_expansion_multiplier": 4,
    "loss_binary_factor": 5.0,
    "dropout_all": False            # Dropout all embedding dimensions in the Transformer block and the initial GPTModel.drop_emb including the binary number part
}

tokenizer.setConfig(GPT_CONFIG_124M)

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M, tokenizer)
print(f"model.CFG: {model.CFG}")
print(f"Model parameter count: {round(model.countParameters()/1000000)} M    ({model.countParameters()}) ")
if args.load_model:
    if os.path.isfile(args.model_path):
        print(f"Loading model from {args.model_path} ...")
        model.load_state_dict(torch.load(args.model_path, weights_only=False))
    else:
        print(f"WARNING Could not find and load model file {args.model_path} ...")


model.to(device)
model.eval()

if args.run_mode == "chat-simple":
    print("Running chat simple mode")

    user_input = input("Enter a sentence to chat ('q' to quit)> ")
    while user_input != "q":
        chat_input = f"<prompt> {user_input} </prompt>"
        response = model.generateResponseSimple(device, chat_input)
        print(f"Bot: {response}\n")
        user_input = input("> ")

    print(f"Bot: Goodbye :)\n")
    exit(0)


print(f"--- Test model before training: device: {device} ---")
# generate_and_print_sample(model, device, default_start_context)
model.generateAndPrintSample(device, default_start_context)
print(f"--------------------------------------")

train_loader = create_loader_pretraining(tokenizer, resource_uri=args.train_uri, name=args.dataset_name, text_key=args.dataset_key,
                                         split=args.dataset_training_split, records_to_process=args.records_to_process,
                                         batch_size=args.batch_size, max_length=model.CFG["context_length"], stride=model.CFG["context_length"],
                                         drop_last=False, shuffle=False, num_workers=args.num_workers)

validation_loader = create_loader_pretraining(tokenizer, resource_uri=args.validation_uri, name=args.dataset_name, text_key=args.dataset_key,
                                              split=args.dataset_validation_split, records_to_process=args.records_to_process,
                                              batch_size=args.batch_size, max_length=model.CFG["context_length"], stride=model.CFG["context_length"],
                                              drop_last=False, shuffle=False, num_workers=args.num_workers)

train_loader.dataset.debugPrintText(args.dbg_print_text)
train_loader.dataset.writeTextToDebugFile(args.dbg_write_records_to_file)
train_loader.dataset.setDebugDataFileName(args.dbg_records_file_name)

# validation_loader = create_loader_foundation(tokenizer, args.validation_uri, args.records_to_process, batch_size=args.batch_size, max_length=model.CFG["context_length"], stride=model.CFG["context_length"], drop_last=False, shuffle=False, num_workers=0)

# -------------------------------------------------------
# --- Print initial loos before training if requested ---
# -------------------------------------------------------
if args.print_initial_loss:
    print("--- Calculating initial model loss ---")
    model.to(device)
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=args.eval_batches)
        val_loss = calc_loss_loader(validation_loader, model, device, num_batches=args.eval_batches)
    print(f"Training loss   : {train_loss}")
    print(f"Validation loss : {val_loss}")
    print("---------------------------------")


# torch.manual_seed(123)
# model = GPTModel(GPT_CONFIG_124M, tokenizer)
model.to(device)
optimizer = torch.optim.AdamW(
     model.parameters(),
    lr=0.0004, weight_decay=0.1
)

# *********************************
# *********************************
# *** TODO: Foundation Traniner ***
# *********************************
# *********************************
# trainer = FoundationTrainer (model=model, train_loader=train_loader, val_loader=validation_loader, optimizer=optimizer, device=device)
# trainer.eval_freq = args.eval_freq
# trainer.eval_iter = args.eval_batches
# train_losses, val_losses, tokens_seen = trainer.trainModel(num_epochs=num_epochs, start_context=default_start_context)

train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, validation_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=args.eval_freq, eval_iter=args.eval_batches,
    start_context=default_start_context
)

# ---------------------------
# --- Plotting the losses ---
# ---------------------------
if args.plot:
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

# ---------------------------------
# --- generate some output text ---
# ---------------------------------
## device = "cpu"
model.to(device)
model.eval()


torch.manual_seed(123)
token_ids = model.generateText(
    idx=model.textToTokenIds(default_start_context).to(device),
    max_new_tokens=15,
    context_size=model.CFG["context_length"],
    top_k=25,
    temperature=1.4
)
print("[Advanced] Output text:\n", model.tokenIdsToText(token_ids))

# --------------------------
# --- Saving the weights ---
# --------------------------
if args.save_model:
    print(f"Saving model to {args.model_path} ...")
    model.to("cpu")
    torch.save(model.state_dict(), args.model_path)
    model.to(device)
    print(f"Done!")

#
# torch.save({
#     "model_state_dict": model.state_dict(),
#     "optimizer_state_dict": optimizer.state_dict(),
#     },
#     "model_and_optimizer.pth"
# )


