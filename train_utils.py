import json 
import os 
import dill
from datasets import load_dataset
import ast
import re
import pandas as pd
from copy import copy

ACCEPTED_MODELS = ["llama2", "mistral7b", "gpt-4o"]

def compare_results(current_best, prospect):
    
    if prospect["accuracy_score"] > current_best["accuracy_score"]:
        return True
    
    if prospect["accuracy_score"] == current_best["accuracy_score"] and prospect["avg_f1_score"] > current_best["avg_f1_score"]:
        return True
    
    return False

def indicator_func(gold_size, indicated_val=0):
    return [indicated_val] * gold_size

def index_value(index, gold):
    return 1 if 0 <= index < len(gold) else 0

def existence_value(index, gold):
    return 1 if index in gold else 0

def f1_score(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def precision_score(tp, fp):
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)

def recall_score(tp, fn):
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)

def partial_recall_precision(input_list, gold):
    precision = -1
    recall = -1
    
    true_positive = []
    false_positive = []
    out_of_order = []
    false_negative = copy(gold)
    error_dict = {
        "fp": false_positive,
        "fn": false_negative,
        "ooo": out_of_order
    }
    
    if input_list == None:
        return 0, 0, 0, error_dict
    
    if len(gold) == 0 and len(input_list) == 0:
        #precision, recall
        return 1, 1, 1, error_dict

    for i, index in enumerate(input_list):
        
        if i >= len(gold) or input_list[i] != gold[i]:
            out_of_order.append(index)
            
        if index in gold and index in false_negative:
            true_positive.append(index)
            false_negative.remove(index)
        else:
            false_positive.append(index)
            
    precision = precision_score(len(true_positive), len(false_positive))
    recall = recall_score(len(true_positive), len(false_negative))
    
    return precision, recall, f1_score(precision, recall), error_dict

ACCEPTED_ALGORITHMS = {
    "bfs": {
        "name": "Breadth-first Search",
        "output_prefix": "Reachable Nodes:",
        "output_regex": r"\s*(\[.*?\])",
        "precision_recall": partial_recall_precision,
        "partial_eval": index_value
    },
    "dfs": {
        "name": "Depth-first Search",
        "output_prefix": "Connected Components:",
        "output_regex": r"\s*(\[.*?\])", #r"\s*(\[(\[(?:\d+(?:,\s*\d+)*)\](?:,\s*\[(?:\d+(?:,\s*\d+)*)\])*)\]|\[(\((?:\d+(?:,\s*\d+)*)\)(?:,\s*\((?:\d+(?:,\s*\d+)*)\))*)\])",#r"\s*(\[\[(?:\d+(?:,\s*)?)+\](?:,\s*\[(?:\d+(?:,\s*)?)+\])*\])",
        "malformed_ouput_regex":r"\s*(\[\[.*?\]\])",
        "precision_recall": partial_recall_precision,
        "partial_eval": existence_value
    },
    "dijkstra": {
        "name": "Dijkstra's",
        "output_prefix": "Distances:",
        "output_regex": r"\s*(\[.*?\])",
        "precision_recall": partial_recall_precision,
        "partial_eval": existence_value
    },
    "floyd_warshall": {
        "name": "Floyd-Warshall",
        "output_prefix": "Distances:",
        "output_regex": r"\s*(\[.*?\])",
        "precision_recall": partial_recall_precision,
        "partial_eval": existence_value
    },
    "mst_prim": {
        "name": "Prim MST",
        "output_prefix": "MST Edges:",
        "output_regex": r"\s*(\[.*?\])",
        "precision_recall": partial_recall_precision,
        "partial_eval": existence_value
    }
}

def count_invalid_items(items):
    """
    Counts the number of invalid items in a list.
    An item is considered invalid if it is neither an integer nor a tuple of the format (int, int, float/int).

    Parameters:
    items: The list of items to check.

    Returns:
    int: The number of invalid items.
    """
    def is_valid_item(item):
        """
        Checks if an item is an integer or a tuple of the format (int, int, float/int).

        Parameters:
        item: The item to check.

        Returns:
        bool: True if the item is an integer or a tuple of the format (int, int, float/int), False otherwise.
        """
        if isinstance(item, int):
            return True
        elif isinstance(item, tuple) and len(item) == 3:
            if isinstance(item[0], int) and isinstance(item[1], int) and (isinstance(item[2], (int, float))):
                return True
        return False
    
    invalid_count = 0
    for item in items:
        if not is_valid_item(item):
            invalid_count += 1

    return invalid_count


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
    return [(f"{data_inst['instruction']} {data_inst['input']}", data_inst["output"]) for data_inst in test_dataset]

def log_results(output_dir, run_name, data):
    write_json(os.path.join(output_dir, run_name), data)

def resolve_output_dirs(config, graph_size):
    output_dir = f"{config.data_path}/graph_size_{graph_size}/{config.dataset_type}/{config.model}/{config.output_path}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    return output_dir

def retrieve_output_dirs(config, graph_size):
    output_dir = f"{config.data_path}/graph_size_{graph_size}/{config.dataset_type}/{config.model}/{config.output_path}"
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"{output_dir} does not exist")
    return output_dir

def print_metrics_dict(metrics_dict):
    print("============================")
    for metric,score in metrics_dict.items():
        print(f"{metric}: {score}")
    print("============================")

def answer_only_formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])): 
        output_texts.append(f"{example['instruction'][i]} {example['input'][i]}\n{example['output'][i]}")
    return output_texts

def parse_malformed_string(input_str):
    """
    Parses a malformed string to extract numbers and reconstruct a list of nodes.

    Parameters:
    - input_str: A malformed string containing node information.

    Returns:
    - A list of integers representing nodes extracted from the input string.
    """
    # Use a regular expression to find all numbers in the string
    numbers = re.findall(r'\d+', input_str)

    # Convert found numbers to integers
    node_list = [int(num) for num in numbers]

    return node_list

def unpack_literal(literal_str, gold=False):
    """
    Parses a string containing a literal list of tuples and automatically unpacks the tuples.

    Parameters:
    - literal_str: A string containing a Python literal expression, specifically a list of tuples.

    Returns:
    - A list where each tuple from the original list is unpacked, 
      according to specified logic.
    """
    # Parse the string to a Python literal (safely with ast.literal_eval)
    try:
        parsed_literal = ast.literal_eval(literal_str)
    # except SyntaxError:
    #     parsed_literal = parse_malformed_string(literal_str)
    # except ValueError:
    #     return None
    except:
        return None
    
    # Initialize an empty list to store unpacked results
    unpacked_list = []
    
    # Iterate through the parsed list, unpacking each tuple
    for item in parsed_literal:
        if gold and isinstance(item,list):
            item = tuple(item)
        unpacked_list.append(item)

    return unpacked_list

def list_to_indicator(algorithm, input_list, gold_size):
    """
    Converts a list of indices to a binary indicator list of a given size.
    
    Parameters:
    - input_list: List of indices.
    - graph_size: The size of the graph or the desired length of the output list.
    
    Returns:
    - A list of size `graph_size` with 1s at indices specified in `input_list`
      and 0s elsewhere.
    """
    # Initialize the indicator list with 0s
    
    indicator_list = indicator_func(gold_size)
    
    # Set 1 at each index present in input_list
    for index in input_list:
        try:
            indicator_list[int(index)] = ACCEPTED_ALGORITHMS[algorithm]["partial_eval"](index, gold_size)
        except:
            print(input_list)
            print(index, int(index), isinstance(int(index), str), gold_size)
            raise ValueError
    return indicator_list

def find_last_match(pattern, text):
    matches = list(re.finditer(pattern, text, re.IGNORECASE))
    return matches[-1] if matches else None

def save_results_to_csv(data, output_path):
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)

def save_results_to_excel(data, output_path):
    df = pd.DataFrame(data)
    df.to_excel(output_path, index=False)

def save_results_to_latex(data, output_path):
    df = pd.DataFrame(data)
    with open(output_path, 'w') as f:
        f.write(df.to_latex(index=False))
        
class IncorrectFormatError(Exception):
    """Custom exception for handling incorrect format errors."""
    def __init__(self, message="Input format is incorrect"):
        self.message = message
        super().__init__(self.message)