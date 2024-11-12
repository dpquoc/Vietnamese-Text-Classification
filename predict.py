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

tokenizer = AutoTokenizer.from_pretrained(cfg.TOKENIZER)
model = AutoModelForSequenceClassification.from_pretrained(f'trained_model') # change your dir model

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
    compute_metrics = compute_metrics,
#     optimizers=(get_optimizer(model), None)
    #callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)


test_data = pd.read_csv(f'data/test.csv')
test_dataset = Dataset.from_pandas(test_data)
test_tokenized_dataset = test_dataset.map(preprocess, remove_columns=['id', 'content', 'index_spans', 'toxic'])

test_predictions = trainer.predict(test_tokenized_dataset)
res = compute_metrics(test_predictions)
print('Accuracy: ', res['accuracy'])
print('F1: ', res['f1'])

