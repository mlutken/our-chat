import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import terminedia as TM

import numpy as np
# from functorch.dim.reference import positional

from torch import Tensor
from typing import Optional
from nndataloaders import *

from nnplotting import *
from system_globals import *
from tokenizer_utils import *

def tensorRemoveIndices_(tensor, indices):
    mask = torch.ones(tensor.numel(), dtype=torch.bool)
    mask[indices] = False
    return tensor[mask]

def getNumberIndicesToRemove(idx):
    return tensor[mask]

class EmbeddingWithNumbers(nn.Module):
    def __init__(self, tokenizer, vocab_size: int, embedding_dim: int, num_bits, number_bits_begin, number_bits_end) -> None:
        super().__init__()
        self.number_bits_begin = number_bits_begin
        self.number_bits_end = number_bits_end
        self.tokenizer = tokenizer
        self.embed_dim = embedding_dim
        self.number_bits = num_bits
        self.standard_dim = embedding_dim   # Perhaps use: embedding_dim - number_bits to optimize
        ### self.standard_dim = embedding_dim - number_bits   # Perhaps use: embedding_dim - number_bits to optimize
        self.standard_embedding = nn.Embedding(vocab_size, self.standard_dim)
        # self.binToDecimalWeights = torch.pow(2, torch.arange(num_bits + 1))

    def binaryTensorToDecimal(self, binary_tensor):
        return binaryTensorToDecimal(binary_tensor)

    ## NUMBER_PARSING_FIXME
    def forward(self, token_ids: Tensor) -> Tensor:
        with torch.no_grad():
            processedIds = self.tokenizer.processIdBatchForEmbedding(token_ids)

        # token_ids_copy = processedIds["ids"]
        with torch.no_grad():
            token_ids_copy = processedIds["ids"].to(token_ids.device)
            bin_numbers_tensor = processedIds["bin_numbers"].to(token_ids.device)

        token_ids_copy = token_ids_copy.to(torch.int32)
        # print (f"FIXMENM token_ids_copy: {token_ids_copy}")


        embeddings = self.standard_embedding.forward(token_ids_copy)

        with torch.no_grad():
            embeddings[:,:,self.number_bits_begin:self.number_bits_end] = bin_numbers_tensor[:,:,:]
            ###embeddings[:,:,-self.number_bits:] = bin_numbers_tensor[:,:,:]

        # embeddings[:,:,-self.number_bits:] = bin_numbers_tensor[:,:,:]

        # Set every 16th element to zero
        # embeddings[:,:,11::12] = 0  # Starts at index 15, then every 16th element
        # print (f"FIXMENM token_ids: {token_ids.device}, embeddings: {embeddings.device}, token_ids_copy: {token_ids_copy.device}, token_ids_copy2: {token_ids_copy2.device}, bin_numbers_tensor2: {bin_numbers_tensor2.device}")
        return embeddings



class DropoutBlock(nn.Module):
    def __init__(self, dropout_rate, avoid_begin, avoid_end, dropout_all) -> None:
        super().__init__()
        self.dropout_rate = dropout_rate
        self.avoid_begin = avoid_begin
        self.avoid_end = avoid_end
        self.dropout_all = dropout_all


    def forward(self, x):
        if not self.training or self.dropout_rate == 0:
            return x

        if self.dropout_all:
            return F.dropout(x, p=self.dropout_rate, training=self.training)
        else:
            # Split the tensor into parts to exclude and parts to dropout
            *rest, last_dim = x.shape
            to_drop = x[..., :self.avoid_begin]
            to_keep = x[..., self.avoid_begin:self.avoid_end]

            # Apply dropout only to the part to drop
            dropped = F.dropout(to_drop, p=self.dropout_rate, training=self.training)
            # Concatenate the dropped and kept parts
            return torch.cat([dropped, to_keep], dim=-1)


class LinearBlock(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.d_in = in_features
        self.d_out = out_features

    def forward(self, x):
        return super().forward(x)


# TODO: Need to decide if we want to exclude number bits from dropout here as well. Currently I think we should not, but it's worth remembering that we do not do it here!
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        assert (d_out == d_in), \
            "d_out must be equal to d_in"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )
        # print (f"d_in              : {d_in}")
        # print (f"self.d_out        : {self.d_out}")
        # print (f"self.num_heads    : {self.num_heads}")
        # print (f"self.head_dim     : {self.head_dim}")
        # print (f"self.W_query.shape: {self.W_query.weight.shape}")
        # print (f"self.mask.shape   : {self.mask.shape}")
        # print (f"!!! FIXMENM self.mask\n: {self.mask}")

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        # print (f"* FIXMENM * attn_scores before mask:\n{attn_scores}")

        attn_scores.masked_fill_(mask_bool, -torch.inf)
        # print (f"* FIXMENM * attn_scores after mask:\n{attn_scores}")

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)

        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )
        context_vec = self.out_proj(context_vec)

        # print (f"* self.mask.shape   : {self.mask.shape}")
        # print (f"* mask_bool.shape   : {mask_bool.shape}")
        # print (f"* mask_bool:\n{mask_bool}")
        # print (f"* keys.shape        : {keys.shape}")
        # print (f"* queries.shape     : {queries.shape}")
        # print (f"* values.shape      : {values.shape}")
        # print (f"* mask_bool.shape   : {mask_bool.shape}")
        # print (f"* num_tokens        : {num_tokens}")
        # print (f"* attn_scores.shape : {attn_scores.shape}")
        # print (f"* attn_weights.shape: {attn_weights.shape}")
        # print (f"* keys.shape[-1]    : {keys.shape[-1]}")
        # print (f"* keys.shape        : {keys.shape}")
        # print (f"* context_vec.shape : {context_vec.shape}")

        return context_vec


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        # print(f"FIXMENM ******** LayerNorm.d_pass_through_start: emb_dim: {emb_dim}")
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        # print(f"FIXMENM ******** LayerNorm. x.shape: {x.shape}, self.scale.shape: {self.scale.shape}")
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        expand_multiplier = cfg["feed_forward_layer_expansion_multiplier"]
        linear_layer_dimension = expand_multiplier * cfg["emb_dim"]
        # print(f"FIXMENM ******** FeedForward.expand_multiplier: {expand_multiplier}, emb_dim: {cfg["emb_dim"]}, linear_layer_dimension: {linear_layer_dimension}")
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], linear_layer_dimension),
            GELU(),   # Set to same to disable passthrough as we do it on the FeedForward block level
            nn.Linear(linear_layer_dimension, cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.number_bits = cfg["number_bits"]
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias = cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.dropout = DropoutBlock(cfg["drop_rate"], cfg["avoid_begin"], cfg["avoid_end"], cfg["dropout_all"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + shortcut
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, max_sequence_length, emb_dim, avoid_begin, avoid_end):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.emb_dim = emb_dim
        self.avoid_begin = avoid_begin
        self.avoid_end = avoid_end

        # self.zeroes_PE = torch.zeros(max_sequence_length, emb_dim)
        self.default_PE = self.calcPE()

    def calcPE(self):
        even_i = torch.arange(0, self.emb_dim, 2).float()
        denominator = torch.pow(10000, even_i/self.emb_dim)
        position = (torch.arange(self.max_sequence_length)
                          .reshape(self.max_sequence_length, 1))
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        PE[:,self.avoid_begin:self.avoid_end] = 0
        return PE

    def forward(self, x):
        position_tensor = self.default_PE[:x.shape[-1], :]
        # print(f"FIXMENM PositionalEncoding x.shape[-1]: {x.shape[-1]}, self.default_PE.shape: {self.default_PE.shape}., position_tensor.shape: {position_tensor.shape}")
        # saveTensorToFile("/tmp/_position_tensor", position_tensor)      # FIXMENM
        # saveTensorToFile("/tmp/_position_default_PE", self.default_PE)  # FIXMENM
        return position_tensor
        # return self.default_PE[:x.shape[-1], :]


class GPTModel(nn.Module):
    def __init__(self, cfg, tokenizer):
        super().__init__()

        self.tokenizer = tokenizer
        self.embed_dim = cfg["emb_dim"]
        self.context_length = cfg["context_length"]
        self.drop_rate = cfg["drop_rate"]
        self.vocab_size = cfg["vocab_size"]
        self.n_layers = cfg["n_layers"]
        self.number_bits = cfg["number_bits"]
        self.dropout_all = cfg["dropout_all"]

        self.number_bits_end  = self.embed_dim

        self.number_bits_begin  = self.number_bits_end - self.number_bits
        self.logit_bits_end  = self.number_bits_begin
        self.logit_bits_start  = 0

        self.avoid_begin = self.number_bits_begin
        self.avoid_end = self.number_bits_end

        self.outhead_input_size = self.number_bits_begin

        self.CFG = cfg
        self.CFG["number_bits_begin"] = self.number_bits_begin
        self.CFG["number_bits_end"] = self.number_bits_end
        self.CFG["avoid_begin"] = self.avoid_begin
        self.CFG["avoid_end"] = self.avoid_end

        print(f"FIXMENM A pass_through: [number_bits: [{self.number_bits_begin} : {self.number_bits_end}], outhead_input_size; {self.outhead_input_size}")

        self.tok_emb = EmbeddingWithNumbers(tokenizer, self.vocab_size, self.embed_dim, self.number_bits, self.number_bits_begin, self.number_bits_end)
        # self.tok_emb = CustomEmbedding(self.vocab_size, self.embed_dim)
        ### self.pos_emb = nn.Embedding(self.context_length, self.embed_dim)

        self.position_encoder = PositionalEncoding(self.context_length, self.embed_dim, self.avoid_begin, self.avoid_end)
        self.drop_emb = DropoutBlock(self.drop_rate, self.avoid_begin, self.avoid_end, self.dropout_all)

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(self.CFG) for _ in range(self.n_layers)])

        self.final_norm = LayerNorm(self.embed_dim)
        self.out_head = nn.Linear(
            self.outhead_input_size, self.vocab_size, bias=False
        )
        self.out_head_number_part = nn.Linear(
            self.number_bits, self.number_bits, bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        # print(f"FIXMENM A in_idx.shape: {in_idx}")
        ## print (in_idx)  # FIXMENM
        tok_embeds = self.tok_emb.forward(in_idx)
        # print(f"FIXMENM B in_idx.shape: {in_idx.shape}")

        # with torch.no_grad():
        #     save_pass_through = tok_embeds[:, :, self.d_pass_through_begin:self.d_pass_through_end]

        ### pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        ### x = tok_embeds + pos_embeds.to(in_idx.device)

        position_tensor = self.position_encoder(in_idx).to(in_idx.device)
        # print(f"FIXMENM position_tensor.shape :{position_tensor.shape}")

        x = tok_embeds + position_tensor
        # number_part = x[:, :, -self.number_bits:]
        # print(f"FIXMENM AFTER POSITION encoding number_part[0,8]: {number_part[0,0]}")

        # x = tok_embeds + pos_embeds + self.position_encoder(in_idx).to(in_idx.device)

        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)

        ## x_logits_part = x[..., :-self.number_bits]  # TODO: USe this !!!
        ## x_number_part = x[:, :, -self.number_bits:]
        x_logits_part = x[..., self.logit_bits_start:self.logit_bits_end]  # TODO: USe this !!!
        x_number_part = x[:, :, self.number_bits_begin:self.number_bits_end]

        x_number_part_out = self.out_head_number_part(x_number_part)
        logits = self.out_head(x_logits_part)
        # binary_tensor = (x_number_part > 0.5).int()

        # numbers_vec = binaryTensorToDecimal(x_number_part_out)

        ## FIXMENM x.shape: torch.Size([2, 256, 144]), logits.shape: torch.Size([2, 256, 1396]), x_number_part: torch.Size([2, 256, 8])
        # print(f"FIXMENM numbers_vec :{numbers_vec}")
        # print(f"FIXMENM x.shape :{x.shape}, logits.shape :{logits.shape}, x_number_part: {x_number_part.shape}, x_number_part_out: {x_number_part_out.shape}")
        # print(f"FIXMENM x_number_part[0,0]: {x_number_part[0,0]}, binary_tensor[0,0]: {binary_tensor[0,0]}")
        ## print(f"FIXMENM logits.shape :{logits.shape}")
        ## print(f"FIXMENM position_encoder.shape:{self.position_encoder().shape}")
        ## print(f"")
        return logits, x_number_part_out

    ## NUMBER_PARSING_FIXME
    def calcLossBatch(self, input_batch, target_batch, device):
        # print(f" FIXMENM A target_batch: {target_batch}")

        with torch.no_grad():
            processedIds = self.tokenizer.processIdBatchForEmbedding(target_batch)

        # saveTensorToFile("/tmp/_target_batch_orig", target_batch)    # FIXMENM
        # saveTensorToFile("/tmp/_target_batch_processed", processedIds["ids"])    # FIXMENM

        with torch.no_grad():
            target_batch = processedIds["ids"].to(device)
            target_binary_numbers_vec = processedIds["bin_numbers"].to(device)
            batch_size = int(target_batch.size(0))
            tokens_size = int(target_batch.size(1))

        # print(f" FIXMENM B target_batch: {target_batch}")

        target_batch = target_batch.to(torch.int64)

        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)
        # --- Run neural network ---
        # print(f"FIXMENM target_batch.shape: {target_batch.shape}")
        # print(f"FIXMENM target_batch({target_batch.shape}): {target_batch}")

        logits, binary_numbers_part = self.forward(input_batch)

        binary_numbers_diff = target_binary_numbers_vec - binary_numbers_part
        binary_numbers_diff2 = binary_numbers_diff ** 2
        # binary_numbers_diff2_sum = binary_numbers_diff2.sum(-1)

        # numbers_vec = binaryTensorToDecimal(binary_numbers_part)
        # numbers_vec_diff = target_numbers_vec - numbers_vec
        # numbers_vec_diff2 = numbers_vec_diff * numbers_vec_diff
        # numbers_vec_diff2_sum = numbers_vec_diff2.sum(-1)

        target_batch_one_hot = torch.zeros(batch_size, tokens_size, self.vocab_size).to(device)

        # saveTensorToFile("/tmp/_target_scatter_onme_hot", target_batch.to(torch.int32))    # FIXMENM

        target_batch_one_hot.scatter_(2, target_batch.unsqueeze(-1), 1.0)

        # target_batch_one_hot = target_batch_one_hot.flatten(0, 1)
        # target_batch_one_hot.scatter_(1, target_batch.unsqueeze(1), 1.0)

        # loss = torch.nn.functional.cross_entropy(
        #     logits.flatten(0, 1), target_batch.flatten()
        # )
        loss_indices = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), target_batch_one_hot.flatten(0, 1)
        )

        # loss_binary_numbers = torch.nn.functional.cross_entropy(
        #     binary_numbers_part.flatten(0, 1), target_binary_numbers_vec.flatten(0, 1)
        # )

        binary_numbers_diff2_flatten = binary_numbers_diff2.flatten()
        binary_numbers_diff2_flatten_len_sqrt = math.sqrt(binary_numbers_diff2_flatten.shape[-1])
        # binary_numbers_diff2_flatten_len_sqrt = torch.sqrt(binary_numbers_diff2_flatten.shape)
        # loss_binary_numbers = binary_numbers_diff2_flatten.sum(-1).sqrt()
        loss_binary_numbers = binary_numbers_diff2_flatten.sum(-1) / binary_numbers_diff2_flatten_len_sqrt
        loss = loss_indices + loss_binary_numbers
        # print(f"!!FIXMENM logits.shape: {logits.shape}, binary_numbers_diff2.shape: {binary_numbers_diff2.shape}, binary_numbers_diff2_flatten.shape: {binary_numbers_diff2_flatten.shape}")
        # print(f"!!FIXMENM !! binary_numbers_diff2_flatten_len_sqrt: {binary_numbers_diff2_flatten_len_sqrt}")

        # print(f"!!FIXMENM !! loss_indices: {loss_indices}, loss_binary_numbers: {loss_binary_numbers}, loss: {loss}")
        return loss

    def usePassThrough(self, x):
        return self.d_pass_through_begin < x.shape[-1]

    # def generateTextSimple(self, idx, max_new_tokens, context_size):
    #     return self._generateTextSimpleNumberParsing(idx, max_new_tokens, context_size)

    def generateText(self, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
        print (f"FIXMENM Number GT: max_new_tokens : {max_new_tokens}, context_size: {context_size}")
        # print (f"GT: idx : {idx}")
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            with torch.no_grad():
                logits, binary_numbers_part = self.forward(idx_cond)
            logits = logits[:, -1, :]       # Get last element
            if top_k is not None:
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(
                    logits < min_val,
                    torch.tensor(float('-inf')).to(logits.device),
                    logits
                )
            if temperature > 0.0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            if idx_next == eos_id:
                break
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def generateTextSimple(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx = self.generateNextTokenSimple(idx)
            last_token = idx[0][-1]
            if last_token == END_TOKEN_ID or last_token == RESPONSE_END_ID:
                break
        return idx

    ## NUMBER_PARSING_FIXME
    def generateNextTokenSimple(self, idx):
        context_size = self.CFG["context_length"]
        curDevice = idx.device

        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits, binary_numbers_part = self.forward(idx_cond)

        logits = logits[:, -1, :]   # Get last element
        # print (f"FIXMENM logits: {logits}")
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        id_next = idx_next[0,0]
        prev_id = 0
        if idx.shape[1] > 1:
            prev_id = idx[0,-1]
            # print(f"FIXMENM id_next: '{id_next}, prev_id: {prev_id}'")

        # print(f"FIXMENM id_next: '{id_next}, prev_id: {prev_id}'")

        ids_to_append = id_next

        if id_next == VALUE_ID:
            binary_numbers_part = binary_numbers_part[:, -1, :]   # Get last element
            number_tensor = binaryTensorToDecimal(binary_numbers_part)
            number_tensor_concat = number_tensor.unsqueeze(0)

            ids_to_append = number_tensor_concat

            number_id_tensor = torch.tensor(NUMBER_ID, device=curDevice).unsqueeze(0).unsqueeze(0)
            if not idIsNumber(prev_id):
                ids_to_append = torch.cat((number_id_tensor, ids_to_append), dim=1)

            idx = torch.cat((idx, ids_to_append), dim=1)

            # print(f"FIXMENM ## ids_to_append   : {ids_to_append}")
            # print(f"FIXMENM ## number_id_tensor: {number_id_tensor}")

            # if not idIsNumber(prev_id):
            #     print(f"FIXMENM ERROR ** number_id_tensor: {number_id_tensor}, number_tensor_concat: {number_tensor_concat}, prev_id: {prev_id}, id_next: {id_next}")
            #     print(f"FIXMENM ** indices in : {idx_cond}")
            #     print(f"FIXMENM ** indices out: {idx}")

            # idx = torch.cat((idx, number_tensor_concat), dim=1)
            # print(f"FIXMENM ## number_id_tensor: {number_id_tensor}, number_tensor_concat: {number_tensor_concat}, prev_id: {prev_id}, id_next: {id_next}, ids_to_append: {ids_to_append}")
            # if not idIsNumber(prev_id):
            #     print(f"FIXMENM ERROR ** number_id_tensor: {number_id_tensor}, number_tensor_concat: {number_tensor_concat}, prev_id: {prev_id}, id_next: {id_next}")
            #     print(f"FIXMENM ** indices in : {idx_cond}")
            #     print(f"FIXMENM ** indices out: {idx}")

        else:
            idx = torch.cat((idx, idx_next), dim=1)
            # print (f"FIXMENM !! idx_next: {idx_next}")


        # saveTensorToFile("/tmp/_probas", probas)    # FIXMENM
        return idx

    def generateResponseSimple(self, device, start_context):
        self.eval()
        encoded = self.textToTokenIds(start_context).to(device)
        with torch.no_grad():
            token_ids = self.generateTextSimple( idx=encoded, max_new_tokens=50)
        decoded_text = self.tokenIdsToText(token_ids)
        decoded_text = decoded_text.replace("\n", " ")  # FIXME at some point !!!
        self.train()
        return decoded_text

    def generateAndPrintSample(self, device, start_context):
        decoded_text = self.generateResponseSimple(device, start_context)
        print(f"R: {decoded_text}")  # FIXME at some point !!!

    ## NUMBER_PARSING_FIXME
    def textToTokenIds(self, text):
        encoded = self.tokenizer.encode(text, allowed_special={'<|endoftext|>'})
        # print (f"TTTI: text: {text},  encoded.shape : {torch.tensor(encoded).shape},  encoded.unsqueeze(0).shape : {torch.tensor(encoded).unsqueeze(0).shape}")
        encoded_tensor = torch.tensor(encoded).unsqueeze(0)
        return encoded_tensor


    def tokenIdsToText(self, token_ids):
        flat = token_ids.squeeze(0)
        return self.tokenizer.decode(flat.tolist())


def calc_loss_loader(data_loader, model, device, num_batches):
    if data_loader is None:
        return float("nan")

    total_loss = 0.
    actual_number_of_batches = 0
    for input_batch, target_batch in data_loader:
        actual_number_of_batches += 1
        loss = model.calcLossBatch( input_batch, target_batch, device)
        total_loss += loss.item()
        if (num_batches is not None) and (actual_number_of_batches == num_batches):
            break

    if actual_number_of_batches == 0:
        return 0

    return total_loss / actual_number_of_batches


# --- Training ---
def evaluate_model(model, train_loader, val_loader, device, eval_iter,  do_validation_loss):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = -1
        if do_validation_loss:
            val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, device, start_context):
    pass
    # model.eval()
    # context_size = model.CFG["context_length"]
    # encoded = model.textToTokenIds(start_context).to(device)
    # with torch.no_grad():
    #     token_ids = model.generateTextSimple(
    #         idx=encoded,
    #         max_new_tokens=50, context_size=context_size
    #     )
    # decoded_text = model.tokenIdsToText(token_ids)
    # print(decoded_text.replace("\n", ""))  # FIXME at some point !!!
    # model.train()


def train_model_simple(model, train_loader, val_loader,
                       optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

    trainingStopRequested = False
    for epoch in range(num_epochs):
        print(f"----------------------- ")
        print(f"--- Start epoch {epoch} ---")
        print(f"-----------------------")
        if trainingStopRequested:
            break

        model.train()
        current_batch_number = -1
        for input_batch, target_batch in train_loader:
            current_batch_number += 1
            if trainingStopRequested:
                break
            # print(f"FIXMENM train batch[{current_batch_number}] global_step: {global_step}, tokens_seen: {tokens_seen}")
            optimizer.zero_grad()
            loss = model.calcLossBatch( input_batch, target_batch, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            should_evaluate = (global_step % eval_freq) == 0

            if should_evaluate:
                # print(f"******************************************************************************************************")
                # print(f"**** FIXMENM evalutating model START should_evaluate: {should_evaluate} global_step: {global_step} ***")
                # print(f"******************************************************************************************************")
                train_loss, val_loss = evaluate_model( model, train_loader, val_loader, device, eval_iter, True)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"EVALUATE: Epoch {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}"
                )
                print("******************************************************************************************************")

            with TM.keyboard:
                if (pressed := TM.inkey()) == "q":
                    print(f"INFO: Training stop requested!")
                    train_loader.dataset.forceStop()
                    if not val_loader is None:
                        val_loader.dataset.forceStop()
                    trainingStopRequested = True

        model.generateAndPrintSample(device, start_context)
    return train_losses, val_losses, track_tokens_seen
