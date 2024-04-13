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
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

class GenerativeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.model == "llama2":
     
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
                torch_dtype = torch.float16
                
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_path,
                quantization_config=quant_config,
                torch_dtype=torch_dtype,
                cache_dir=config.cache_dir,
                #device_map= {"": config.device}
            )
            # self.model.config.use_cache = False
            # self.model.config.pretraining_tp = 1
            
            # self.tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
            # self.tokenizer.padding = True
            # self.tokenizer.truncation = True
            # self.tokenizer.padding_side = 'right'
            # self.tokenizer.pad_token = self.tokenizer.eos_token if self.tokenizer.pad_token is None else self.tokenizer.pad_token
            # self.tokenizer.add_tokens(["[INST]"])
            # self.tokenizer.add_special_tokens(dict(eos_token="[/INST]"))
            # self.model.resize_token_embeddings(len(self.tokenizer))
            # self.model.config.eos_token_id = self.tokenizer.eos_token_id
            
            # self.model.to(config.device)
            # self.model.config.use_cache = False
            
        else:
            raise NotImplementedError("Attempting to use a model that hasn't been implemented yet!")
    
    # collate function - to transform list of dictionaries [ {input_ids: [123, ..]}, {.. ] to single batch dictionary { input_ids: [..], labels: [..], attention_mask: [..] }
    def collate(self, elements):
        tokenlist=[e["input_ids"] for e in elements]
        tokens_maxlen=max([len(t) for t in tokenlist])  # length of longest input

        input_ids,labels,attention_masks = [],[],[]
        for tokens in tokenlist:
            # how many pad tokens to add for this sample
            pad_len=tokens_maxlen-len(tokens)

            # pad input_ids with pad_token, labels with ignore_index (-100) and set attention_mask 1 where content, otherwise 0
            input_ids.append( tokens + [self.tokenizer.pad_token_id]*pad_len )   
            labels.append( tokens + [-100]*pad_len )    
            attention_masks.append( [1]*len(tokens) + [0]*pad_len ) 

        batch={
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attention_masks)
        }
        return batch
    
    def tokenize(self, element):
        return self.tokenizer(
            element,
            truncation=True,
            max_length=2048,
            add_special_tokens=False,
        )