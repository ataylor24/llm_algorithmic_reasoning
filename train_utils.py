import json 
import os 
import dill
from datasets import load_dataset

def load_data(data_format, filepath, window):
    if data_format == "json":
        return load_dataset(data_format, data_files={window: [filepath]}, field='text')
    elif data_format == ".pkl":
        return load_dill(filepath)
    else:
        raise NotImplementedError(f"Handling for data format {data_format} not been implemented.")

def load_dill(data_path):
    return dill.load(open(data_path, 'rb'))

def load_json(filepath):
    return json.load(open(filepath, 'r'))

def write_json(outfile, data):
    json.dump(data, open(outfile + ".json", "w"))

def preprocess_test_data(test_dataset):
    return [data_inst["text"].split("\n\n") for data_inst in test_dataset]

def log_results(output_dir, run_name, data):
    write_json(os.path.join(output_dir, run_name), data)

def resolve_output_dirs(config, graph_size):
    output_dir = f"{config.data_path}/graph_size_{graph_size}/{config.dataset_type}/{config.model}/{config.output_path}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    return output_dir

# def answer_only_formatting_prompts_func(example):
#     output_texts = []

#     for i in range(len(example['algorithm_name'])):
#         input_ = f"Please perform the {example['algorithm_name'][0][i]} algorithm for {example['algorithm_goal'][0][i]}. Edgelist: [{example['edgelist'][0][i]}]; Source Node: {str(example['source_node'][0][i])}."
#         output = f"{example['outputs'][0][i]}"
#         if "hints" in example:
#             input_ = input_ + "\n" + example["hints"][0][i]
#         output_texts.append(f"{input_}\n### Reachable Nodes: {output}")
#     print('OUTPUT', output_texts)
#     return output_texts
# def answer_only_formatting_prompts_func(example):
#     output_texts = []
#     for i in range(len(example['algorithm_name'])):
#         input_ = f"Please perform the {example['algorithm_name'][i]} algorithm for {example['algorithm_goal'][i]}. Edgelist: [{example['edgelist'][i]}]; Source Node: {str(example['source_node'][i])}."
#         output = f"{example['outputs'][i]}"
#         if "hints" in example:
#             input_ = input_ + "\n" + example["hints"][i]
#         output_texts.append(f"{input_}\n### Reachable Nodes: [{output}]")
#     return output_texts
def answer_only_formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])): 
        output_texts.append(f"{example['instruction'][i]}\n### Reachable Nodes: [{example['output'][i]}]")
    return output_texts