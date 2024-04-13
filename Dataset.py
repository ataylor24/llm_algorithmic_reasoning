from torch.utils.data import Dataset
import train_utils
import data_generation.data_utils
import datasets
import os
import torch

class AlgReasoningDataset(Dataset):
    def __init__(self, config, dataset, window):
        self.data = train_utils.load_data(config.data_format, os.path.join(os.path.join(config.data_path, config.dataset_type), dataset), window)
        self.window = window
        
    def __len__(self):
        return len(self.data[self.window])

    def __getitem__(self, idx):
        return self.data[self.window][idx]

class LlamaDataset(AlgReasoningDataset):
    def __init__(self, config, dataset, window):
        super().__init__(config, dataset, window)
        
        
    def tokenize_data(self, tokenizer):
        def tokenize(element):
            return tokenizer(
                element["text"],
                padding="max_length",
                truncation=True,
                max_length=4096,
                add_special_tokens=False,
            )
        
        self.data[self.window] = self.data[self.window].map(
            tokenize, 
            batched=True, 
            num_proc=os.cpu_count(),    # multithreaded
            remove_columns=["text"]     # don't need the strings anymore, we have tokens from here on
        )