import logging
import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from train_utils import ACCEPTED_MODELS

class GenerativeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.model in ACCEPTED_MODELS:
     
            if config.use_peft:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    # bnb_4bit_use_double_quant=False,
                )
                torch_dtype = torch.bfloat16
            else:
                quant_config = None
                torch_dtype = torch.float
                
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_path,
                quantization_config=quant_config,
                torch_dtype=torch_dtype,
                cache_dir=config.cache_dir,
            )
            
        else:
            raise NotImplementedError("Attempting to use a model that hasn't been implemented yet!")
