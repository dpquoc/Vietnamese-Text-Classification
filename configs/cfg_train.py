import os
import sys
import json
from types import SimpleNamespace

cfg_train = SimpleNamespace(**{})
cfg_train.learning_rate = 3e-5
cfg_train.train_batch_size = 8
cfg_train.eval_batch_size = 8
cfg_train.epochs=2
cfg_train.weight_decay=0.01
cfg_train.fp16=True
cfg_train.eval_strategy='steps'
cfg_train.save_strategy="steps"
cfg_train.logging_steps=150
cfg_train.eval_steps=150
cfg_train.save_steps=150
cfg_train.lr_scheduler_type='constant'

