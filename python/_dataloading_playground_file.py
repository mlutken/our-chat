from nndataloaders import *
from dictionary_tokenizer import *

print(f"Data loading playground 3")

tokenizer = DictionaryTokenizer("../dictionary")

vocab_size = tokenizer.vocabSize()
batch_size = 12
emb_dim = 144
train_file = "../training_data/math-training-simple-2.txt"

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

# with open(train_file, "r", encoding="utf-8") as file:
#     train_data = file.read()
#

tokenizer.setConfig(CFG)

train_loader = create_iter_loader_TextFile_PreTrain(
    tokenizer,
    train_file,
    batch_size=batch_size,
    max_length=CFG["context_length"],
    stride=CFG["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

print (train_loader)
num_batches = 0
# for input_batch in train_loader:
for input_batch, target_batch in train_loader:
    num_batches += 1
    print(f"input_batch[0][0:3] : {input_batch[0][0:4]},  target_batch[0][0:3]: {target_batch[0][0:3]}")
    print(f"input_batch[-1][0:3]: {input_batch[-1][0:4]}, target_batch[-1][0:3]: {target_batch[-1][0:3]}")
    # print(f"input_batch.shape : {len(input_batch)}")
    print(f"input_batch.shape : {input_batch.shape}, target_batch.shape : {target_batch.shape}")
    # print(f"input_batch[0][0:3] : {input_batch[0][0:4]}")
    # print(f"input_batch[-1][0:3]: {input_batch[-1][0:4]}")
    # print(f"input_batch: {input_batch}")
    # print(f"len(input_batch): {len(input_batch)}")
    # if num_batches <= 1:
    #     print(f"input_batch: {input_batch}")

print(f"num_batches: {num_batches}")




# train_loader = create_data_loader_TextFile_Foundation(
#     tokenizer,
#     train_file,
#     batch_size=batch_size,
#     stride=CFG["context_length"],
#     max_length=CFG["context_length"],
#     drop_last=True,
#     shuffle=True,
#     num_workers=0
# )
#
# print (train_loader)
# num_batches = 0
# for input_batch, target_batch in train_loader:
#     num_batches += 1
#     if num_batches <= 1:
#         print(f"input_batch: {input_batch}")
#
# print(f"num_batches: {num_batches}")

# token_queue_ = collections.deque([], maxlen=1000)
# token_queue_.appendleft(1)
# token_queue_.appendleft(2)
# token_queue_.appendleft(3)
# token_queue_.appendleft(4)
#
# print(f"A len(token_queue_): {len(token_queue_)}")
# print(f"token_queue_.pop(): {token_queue_.pop()}")
# print(f"token_queue_.pop(): {token_queue_.pop()}")
# print(f"B len(token_queue_): {len(token_queue_)}")



