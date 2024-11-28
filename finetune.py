import os
import torch

from datasets import load_dataset
from huggingface_hub import login
from esm.models.esm3 import ESM3
from trl import CPOConfig
from trl.trainer.utils import DPODataCollatorWithPadding
from utils import ESMCPOTrainer, ESMDataCollator
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

login()
model = ESM3.from_pretrained("esm3-open")
train_dataset = load_dataset("csv", data_files={"data/dpo_small.csv"})

config = CPOConfig(
    max_length=512, 
    output_dir="weights", 
    remove_unused_columns=False
)
trainer = ESMCPOTrainer(
    model=model,
    args=config,
    train_dataset=train_dataset["train"],
    data_collator=DPODataCollatorWithPadding(pad_token_id=1),
    processing_class=EsmSequenceTokenizer()
)

train_dataloader = trainer.get_train_dataloader()
batch = next(iter(train_dataloader))

outputs = trainer.concatenated_forward(model=model, batch=batch)