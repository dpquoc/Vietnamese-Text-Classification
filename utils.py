import torch
from dataclasses import dataclass
from typing import Optional, Union
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy

from configs.cfg_project import cfg
tokenizer = AutoTokenizer.from_pretrained(cfg.TOKENIZER)

def preprocess(example):
    # Tokenize the 'content' (text) column
    tokenized_example = tokenizer("<s> " + example['content'] + "</s>", truncation=True, max_length=128, padding='max_length')
    
    # Convert the 'toxic' column (True/False) into integer labels (1/0)
    tokenized_example['label'] = int(example['toxic'])
    
    return tokenized_example


@dataclass
class DataCollatorForClassification:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, input_batch):
        # Extract labels from the input batch
        labels = [example.pop('label') for example in input_batch]

        # Tokenizer padding (make sure all sequences are the same length)
        batch = self.tokenizer.pad(
            input_batch,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt'  # Return tensors (PyTorch format)
        )

        # Add labels to the batch
        batch['labels'] = torch.tensor(labels, dtype=torch.int64)
        
        return batch