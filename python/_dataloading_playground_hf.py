from nndataloaders import *
from dictionary_tokenizer import *

print(f"Data loading playground 4")

tokenizer = DictionaryTokenizer("../dictionary")

vocab_size = tokenizer.vocabSize()
batch_size = 12
emb_dim = 144



# g_name = "en"

# g_name = "CC-MAIN-2025-26"

# g_hugging_face_uri = "hf:HuggingFaceFW/finewiki"
# g_hugging_face_uri = "hf:HuggingFaceFW/fineweb"
g_hugging_face_uri = "hf:Fredithefish/Instruction-Tuning-with-GPT-4-RedPajama-Chat"
g_dl_data = dataloader_lookup(g_hugging_face_uri)

g_name = g_dl_data["dataset_name"]
g_dl_pre_process = g_dl_data["pre_process"]
g_text_key = g_dl_data["dataset_key"]

# FIXMENM BEGIN
# g_dl_pre_process = ProcessHumanBot1()
# FIXMENM END

CFG = {
    "vocab_size": vocab_size,
    "context_length": 256,
    "emb_dim": emb_dim, # 768
    "n_heads": 6,   # 12
    "n_layers": 6,  # 12
    "drop_rate": 0.1,
    "qkv_bias": False,
    "number_bits": 32,    #
    "feed_forward_layer_expansion_multiplier": 4,
    "loss_binary_factor": 5.0
}

g_records_to_process = 1000
g_split = 'train'

tokenizer.setConfig(CFG)

train_loader = create_loader_pretraining(
    tokenizer,
    g_hugging_face_uri,
    name=g_name,
    text_key=g_text_key,
    split=g_split,
    records_to_process=g_records_to_process,
    batch_size=batch_size,
    max_length=CFG["context_length"],
    stride=CFG["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

train_loader.dataset.writeTextToDebugFile(True)
train_loader.dataset.processCallbackSet(g_dl_pre_process)

num_batches = 0
# for input_batch in train_loader:
for input_batch, target_batch in train_loader:
    num_batches += 1
    print(f"input_batch[0][0:3] : {input_batch[0][0:4]},  target_batch[0][0:3]: {target_batch[0][0:3]}")
    # print(f"input_batch[-1][0:3]: {input_batch[-1][0:4]}, target_batch[-1][0:3]: {target_batch[-1][0:3]}")
    # print(f"input_batch.shape : {len(input_batch)}")
    # print(f"input_batch.shape : {input_batch.shape}, target_batch.shape : {target_batch.shape}")
    # print(f"input_batch[0][0:3] : {input_batch[0][0:4]}")
    # print(f"input_batch[-1][0:3]: {input_batch[-1][0:4]}")
    # print(f"input_batch: {input_batch}")
    # print(f"len(input_batch): {len(input_batch)}")
    # if num_batches <= 1:
    #     print(f"input_batch: {input_batch}")

print(f"FIXMENM num_batches: {num_batches}")


