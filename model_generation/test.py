from pytorch_transformers import GPT2Model, GPT2Tokenizer
import numpy as np
import torch
import math
import torch.nn as nn

sequence_length = 3

model = GPT2Model.from_pretrained("gpt2")

input_ids = torch.tensor(np.zeros(sequence_length), dtype=torch.long)
position_ids = torch.tensor(np.arange(sequence_length).astype(np.float), dtype=torch.long)

# Output of the embeddings addition
embeddings = model.wpe(position_ids) + model.wte(input_ids)

# Output of the first Attention LayerNorm layer
ln_1 = model.h[0].ln_1(embeddings)

# Output of the attention dense layer for Q, K, V
c_attn = model.h[0].attn.c_attn(ln_1).reshape((-1, sequence_length, 2304))

# Splitting the QKV vector
query, key, value = c_attn.split(model.h[0].attn.split_size, dim=2)

# Splitting the heads
split_query, split_key, split_value = model.h[0].attn.split_heads(query), model.h[0].attn.split_heads(key, k=True), model.h[0].attn.split_heads(value)

# QK Matmul
w = torch.matmul(split_query, split_key)

# Scaled QK
scaled_w = w / math.sqrt(split_value.size(-1))

# Scaling the bias with whatever
nd, ns = w.size(-2), w.size(-1)
b = model.h[0].attn.bias[:, :, ns - nd:ns, :ns]
total_scaled_w = scaled_w * b - 1e4 * (1 - b)

# Softmaxing
soft_maxed_w = nn.Softmax(dim=-1)(total_scaled_w)

# Finishing up the attention
attention = torch.matmul(soft_maxed_w, split_value)

# Merge - transpose
merged = attention.permute(0, 2, 1, 3).contiguous()

# Reshaping the merge
size = merged.size()[:-2] + (merged.size(-2) * merged.size(-1),)
fully_merged = merged.view(*size)

# c_proj Conv1D layer
c_proj = model.h[0].attn.c_proj(fully_merged).reshape((-1, sequence_length, 768))

# Addition
xa = embeddings + c_proj

# Second block layer norm
ln_2 = model.h[0].ln_2(xa)

# MLP conf_fc
mlp_conv_fc = model.h[0].mlp.c_fc(ln_2)

# MLP Gelu
mlp_gelu = model.h[0].mlp.act(mlp_conv_fc)

# MLP conf_proj
mlp_conv_proj = model.h[0].mlp.c_proj(mlp_gelu)

# Addition
xm = xa + mlp_conv_proj



print(attention)

