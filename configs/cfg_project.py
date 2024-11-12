import os
import sys
import json
from types import SimpleNamespace


cfg = SimpleNamespace(**{})
cfg.USE_PEFT = False
cfg.FREEZE_LAYERS = 0 # NUMBER OF LAYERS TO FREEZE , DEBERTA LARGE HAS TOTAL OF 24 LAYERS
cfg.FREEZE_EMBEDDINGS = True # BOOLEAN TO FREEZE EMBEDDINGS
cfg.MAX_INPUT = 218 # LENGTH OF CONTEXT PLUS QUESTION ANSWER
cfg.MODEL = 'vinai/phobert-base-v2' 
cfg.TOKENIZER = 'tokenizer'