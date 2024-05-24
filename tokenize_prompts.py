import os
import argparse
import train_utils
from Model import GenerativeModel
from Dataset import AlgReasoningDataset, LlamaDataset
import torch
import re
from datetime import datetime
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
from transformers import (
    TrainingArguments,
    pipeline,
    logging,
    Trainer, 
    DataCollatorForLanguageModeling,
    AutoTokenizer
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer


os.environ["TOKENIZERS_PARALLELISM"] = "false"

def _load_data(graph_size):
    training_datapath = os.path.join(config.hint_level, "training" + "." + config.data_format)
    validation_datapath = os.path.join(config.hint_level, "validation" + "." + config.data_format)
    testing_datapath = os.path.join(config.hint_level, "testing" + "." + config.data_format)
    
    dataloader = load_dataset(f"{config.data_path}/graph_size_{graph_size}/{config.dataset_type}/{config.model}/{config.hint_level}")
    # if config.model == "llama2":
    #     # dataloader = load_dataset("/local2/ataylor2/algorithmic_reasoning/bfs/graph_size_4/llm_data/llama2/with_hints")
    #     dataloader = load_dataset(f"{config.data_path}/graph_size_{graph_size}/{config.dataset_type}/{config.model}/{config.hint_level}")
    # else:
    #     training_dataloader = DataLoader(AlgReasoningDataset(config, training_datapath))
    #     validation_dataloader = DataLoader(AlgReasoningDataset(config, validation_datapath))
    #     testing_dataloader = DataLoader(AlgReasoningDataset(config, testing_datapath))
    #     dataloader = (training_dataloader, validation_dataloader, testing_dataloader)
        
    return dataloader

def main(graph_size):
    # start a new wandb run to track this script
    

    tokenizer_kwargs = {
        "padding": True,
        "truncation": True,
        "cache_dir": config.cache_dir,
        "use_fast": True,
        "token": None,
        # truncate from left, remove some input if too long
        "truncation_side": 'left',
        "padding_side": 'right',
        "model_max_length": config.model_max_length,
        "max_seq_length":512
    }
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, **tokenizer_kwargs)
    # Assign padding token for tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
        
    dataset = _load_data(graph_size)
    

    

    for example in dataset["train"]:
        tokenized_example = tokenizer(f"{example['instruction']} {example['input']}\n{example['output']}")

        for token in tokenized_example['input_ids']:
            print(f"Token Id: {token}, Value: '{tokenizer.decode([token])}'")
        
        break
     

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read configuration JSON file')
    parser.add_argument('config', type=str, help='Path to the configuration JSON file')
    parser.add_argument('graph_size', type=int, help='Path to the configuration JSON file')
    args = parser.parse_args()
    
    global config
    config = train_utils.load_json(args.config)
    config.update(args.__dict__)
    config = argparse.Namespace(**config)
    main(args.graph_size)