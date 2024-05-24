import json 
import os 
import dill
from datasets import load_dataset
import ast
import re

ACCEPTED_MODELS = ["llama2", "mistral7b"]

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

def unpack_literal(literal_str):
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
    except SyntaxError:
        parsed_literal = parse_malformed_string(literal_str)
    except ValueError:
        return None
    
    # Initialize an empty list to store unpacked results
    unpacked_list = []
    
    # Iterate through the parsed list, unpacking each tuple
    for item in parsed_literal:
        if isinstance(item, tuple):
            # For this example, we simply extend the unpacked list with the tuple elements
            unpacked_list.extend(item)
        else:
            # If the item is not a tuple, just append it to the result list
            unpacked_list.append(item)
    
    return unpacked_list


def list_to_indicator(input_list, graph_size):
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
    indicator_list = [0] * graph_size
    
    # Set 1 at each index present in input_list
    for index in input_list:
        if 0 <= index < graph_size:
            indicator_list[index] = 1
    
    return indicator_list

#MISTRAL HINT
# {%- for message in messages %}
#       {%- if message['role'] == 'system' -%}
#           {{- message['content'] -}}
#       {%- else -%}
#           {%- if message['role'] == 'user' -%}
#               {{-'[INST] ' + message['content'].rstrip() + ' [/INST]'-}}
#           {%- else -%}
#               {{-'' + message['content'] + '</s>' -}}
#           {%- endif -%}
#       {%- endif -%}
#   {%- endfor -%}
#   {%- if add_generation_prompt -%}
#       {{-''-}}
#   {%- endif -%}