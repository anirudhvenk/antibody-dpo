import os
import torch

from config import create_config
from datasets import load_dataset
from huggingface_hub import login
from esm.models.esm3 import ESM3
from trl import CPOConfig
from trl.trainer.utils import DPODataCollatorWithPadding
from utils import ESMCPOTrainer, ESMDataCollator
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from peft import LoraConfig, PeftConfig

# DDP is not working for some reason (cuda internal error)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

login()
model = ESM3.from_pretrained("esm3-open")

config = create_config()

dataset = load_dataset("csv", data_files={"data": config.data_dir})
split_datasets = dataset["data"].train_test_split(test_size=0.1)
train_dataset = split_datasets["train"]
test_dataset = split_datasets["test"]

# Freeze all params except sequence track
for name, param in model.named_parameters():
    if name in [
        "encoder.sequence_embed.weight", 
        "output_heads.sequence_head.0.weight", 
        "output_heads.sequence_head.0.bias",
        "output_heads.sequence_head.2.weight",
        "output_heads.sequence_head.2.bias",
        "output_heads.sequence_head.3.weight",
        "output_heads.sequence_head.3.bias"
    ]:
        param.requires_grad = True
    else:
        param.requires_grad = False

config = CPOConfig(
    learning_rate=config.learning_rate,
    per_device_train_batch_size=config.batch_size,
    loss_type=config.loss_type,
    cpo_alpha=config.alpha,
    beta=config.beta,
    output_dir="weights",
    remove_unused_columns=False,
    generate_during_eval=True,
    eval_strategy="steps",
    eval_steps=0.1
)

trainer = ESMCPOTrainer(
    model=model,
    args=config,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=ESMDataCollator(),
    processing_class=EsmSequenceTokenizer()
)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {trainable_params}")

trainer.train()