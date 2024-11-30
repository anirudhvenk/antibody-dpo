import torch

from safetensors.torch import load_file
from esm.models.esm3 import ESM3
from transformers import AutoModel

model = ESM3.from_pretrained("esm3-open")
state_dict = torch.load("weights/checkpoint-1/pytorch_model.bin")
model.load_state_dict(state_dict)
