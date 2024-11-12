import os
import re

from sklearn.model_selection import train_test_split
from typing import Optional, Union
import pandas as pd, numpy as np, torch

from datasets import Dataset
from dataclasses import dataclass

from transformers import AutoTokenizer
from transformers import EarlyStoppingCallback
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import  AutoModelForSequenceClassification, TrainingArguments, Trainer
from torch.optim import AdamW

from utils import preprocess , DataCollatorForClassification
from configs.cfg_project import cfg
from configs.cfg_train import cfg_train
from metrics.metric import compute_metrics



# Load data
train_data = pd.read_csv('data/train.csv')
valid_data = pd.read_csv('data/val.csv')

# Convert to dataset
train_dataset = Dataset.from_pandas(train_data)
valid_dataset = Dataset.from_pandas(valid_data)

# Tokenized dataset
train_tokenized_dataset = train_dataset.map(preprocess, remove_columns=['id', 'content', 'index_spans', 'toxic'])
valid_tokenized_dataset = valid_dataset.map(preprocess, remove_columns=['id', 'content', 'index_spans', 'toxic'])

# Load model
model = AutoModelForSequenceClassification.from_pretrained(cfg.MODEL)
tokenizer = AutoTokenizer.from_pretrained(cfg.TOKENIZER)

if cfg.FREEZE_EMBEDDINGS:
    print('Freezing embeddings.')
#     for param in model.deberta.embeddings.parameters():
    for param in model.roberta.embeddings.parameters():

        param.requires_grad = False
if cfg.FREEZE_LAYERS>0:
    print(f'Freezing {cfg.FREEZE_LAYERS} layers.')
    for layer in model.deberta.encoder.layer[:cfg.FREEZE_LAYERS]:
        for param in layer.parameters():
            param.requires_grad = False


training_args = TrainingArguments(
    learning_rate=cfg_train.learning_rate,  # Set base learning rate for other layers
    per_device_train_batch_size=cfg_train.train_batch_size,
    per_device_eval_batch_size=cfg_train.eval_batch_size,
    num_train_epochs=cfg_train.epochs,
    report_to='none',
    output_dir='./checkpoints',
    overwrite_output_dir=True,
    fp16=cfg_train.fp16,
    gradient_accumulation_steps=1,
    logging_steps=cfg_train.logging_steps,
    eval_strategy=cfg_train.eval_strategy,
    eval_steps=cfg_train.eval_steps,
    save_strategy=cfg_train.save_strategy,
    save_steps=cfg_train.save_steps,
    load_best_model_at_end=False,
    metric_for_best_model='accuracy',
    lr_scheduler_type=cfg_train.lr_scheduler_type,  # Can use 'cosine' or 'linear' based on preference
    weight_decay=cfg_train.weight_decay,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=DataCollatorForClassification(tokenizer=tokenizer),
    train_dataset=train_tokenized_dataset,
    eval_dataset=valid_tokenized_dataset,
    compute_metrics = compute_metrics,
#     optimizers=(get_optimizer(model), None)
    #callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

trainer.train()
trainer.save_model(f'trained_model')

