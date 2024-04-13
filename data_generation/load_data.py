import json 
import os 
import dill
from datasets import load_dataset

def load_data(data_format, filepath):
    if data_format == "json":
        print(json.load(open(filepath)))
        return load_dataset(data_format, data_files=filepath, field='text')
    else:
        raise NotImplementedError(f"Handling for data format {data_format} not been implemented.")
    
load_data("json", "/local2/ataylor2/algorithmic_reasoning/bfs/llm_data/llama2/no_hints/training.json")
