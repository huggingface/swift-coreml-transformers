import torch
from pytorch_transformers import GPT2Model
import numpy as np

sequence_length = 3

input_sequence = torch.tensor(np.zeros(sequence_length), dtype=torch.long).unsqueeze(0)

GPT2Model.from_pretrained('gpt2')(input_sequence)
