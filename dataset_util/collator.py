import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizerBase
from transformers.models.bart.modeling_bart import BartForConditionalGeneration, shift_tokens_right


@dataclass
class DataCollatorForBartDenoisingLM:
    """
    Data collator used for BART denoising language modeling.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data
    """

    tokenizer: PreTrainedTokenizerBase
    decoder_start_token_id: int

    def __call__(self, examples: List[Dict[str, List[int]]]) -> BatchEncoding:
        batch = BatchEncoding(
            {k: torch.LongTensor([examples[i][k] for i in range(len(examples))]) for k, v in examples[0].items()}
        )

        batch["labels"] = batch["input_ids"].clone()
        batch["decoder_input_ids"] = shift_tokens_right(
            batch["labels"], self.tokenizer.pad_token_id, self.decoder_start_token_id
        )

        batch['labels'][batch['labels'] == self.tokenizer.pad_token_id] = -100

        batch["attention_mask"] = (batch["input_ids"] != self.tokenizer.pad_token_id).long()
        batch["decoder_attention_mask"] = (batch["decoder_input_ids"] != self.tokenizer.pad_token_id).long()

        return batch

@dataclass
class DataCollatorForLatentVAE:
    """
    Data collator used for BART denoising language modeling.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data
    """

    tokenizer: PreTrainedTokenizerBase
    # decoder_start_token_id: int

    def __call__(self, examples: List[Dict[str, List[int]]]) -> BatchEncoding:        
        batch = BatchEncoding(
            {k: torch.LongTensor([examples[i][k] for i in range(len(examples))]) for k in examples[0].keys()}
        )
        
        batch["labels"] = batch["input_ids"].clone()
        batch['labels'][batch['labels'] == self.tokenizer.pad_token_id] = -100
        batch["attention_mask"] = (batch["input_ids"] != self.tokenizer.pad_token_id).long()
        
        return batch


@dataclass
class DataCollatorForLatentVAET5:
    """
    Data collator used for BART denoising language modeling.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data
    """

    tokenizer: PreTrainedTokenizerBase
    decoder_start_token_id: int

    def __call__(self, examples: List[Dict[str, List[int]]]) -> BatchEncoding:        
        batch = BatchEncoding(
            {k: torch.LongTensor([examples[i][k] for i in range(len(examples))]) for k in examples[0].keys()}
        )
        
        batch["labels"] = batch["input_ids"].clone() 
        batch["decoder_input_ids"] = shift_tokens_right(batch["labels"], self.tokenizer.pad_token_id, self.decoder_start_token_id)

        batch['labels'][batch['labels'] == self.tokenizer.pad_token_id] = -100
        batch["attention_mask"] = (batch["input_ids"] != self.tokenizer.pad_token_id).long()
        batch["decoder_attention_mask"] = (batch["decoder_input_ids"] != self.tokenizer.pad_token_id).long()
        
        return batch
