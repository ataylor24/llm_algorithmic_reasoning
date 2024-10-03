import json 
import os 
import dill
from datasets import load_dataset
import ast
import re
import pandas as pd
from copy import copy
import networkx as nx
from statistics import median
import numpy as np

def extract_graph_information(prompt, edgelist_pattern):
    source_pattern = r"Source Node:\s*(\d+)"
    source_node = re.search(source_pattern, prompt)
    
    if source_node:
        source_node = int(source_node.group(1))  # Extract the source node and convert to integer
        
    edgelist = parse_string(find_last_match("Edgelist:" + edgelist_pattern, prompt))
   
    return edgelist, source_node

def compute_centrality_statistics(G, weighted=False, connected=True):
    centrality_stats = {}

    # Compute degree centrality
    degree_centrality = nx.degree_centrality(G)
    centrality_stats['degree_mean'] = np.mean(list(degree_centrality.values()))
    centrality_stats['degree_median'] = np.median(list(degree_centrality.values()))
    centrality_stats['degree_max'] = np.max(list(degree_centrality.values()))

    # Compute betweenness centrality
    betweenness_centrality = nx.betweenness_centrality(G, weight='weight' if weighted else None)
    centrality_stats['betweenness_mean'] = np.mean(list(betweenness_centrality.values()))
    centrality_stats['betweenness_median'] = np.median(list(betweenness_centrality.values()))
    centrality_stats['betweenness_max'] = np.max(list(betweenness_centrality.values()))

    # Compute closeness centrality
    closeness_centrality = nx.closeness_centrality(G, distance='weight' if weighted else None)
    centrality_stats['closeness_mean'] = np.mean(list(closeness_centrality.values()))
    centrality_stats['closeness_median'] = np.median(list(closeness_centrality.values()))
    centrality_stats['closeness_max'] = np.max(list(closeness_centrality.values()))

    # if not connected:
    #     return centrality_stats
    
    # # Compute eigenvector centrality
    # eigenvector_centrality = nx.eigenvector_centrality(G, weight='weight' if weighted else None)
    # centrality_stats['eigenvector_mean'] = np.mean(list(eigenvector_centrality.values()))
    # centrality_stats['eigenvector_median'] = np.median(list(eigenvector_centrality.values()))
    # centrality_stats['eigenvector_max'] = np.max(list(eigenvector_centrality.values()))

    return centrality_stats

def load_graph_from_edgelist(edgelist, weighted=False):
    """
    Load a graph from an edge list. If weighted is True, expects (node1, node2, weight) in edgelist.
    """
    if weighted:
        G = nx.Graph()
        G.add_weighted_edges_from(edgelist)  # for weighted graphs
    else:
        G = nx.Graph()
        G.add_edges_from(edgelist)  # for unweighted graphs
    return G

def compute_graph_characteristics(edgelist, weighted=False, source_node=None):
    characteristics = {}

    G = load_graph_from_edgelist(edgelist, weighted=weighted)
    
    # Connected
    characteristics['is_connected'] = nx.is_connected(G)
    
    # Degree Distribution
    degrees = dict(G.degree())
    characteristics['degree_mean'] = sum(degrees.values()) / len(degrees)
    characteristics['degree_median'] = median(list(degrees.values()))
    characteristics['degree_max'] = max(degrees.values())

    # Clustering Coefficient
    avg_clustering = nx.average_clustering(G, weight='weight' if weighted else None)
    characteristics['average_clustering_coefficient'] = avg_clustering

    # Path Length and Diameter
    if characteristics['is_connected']:
        if weighted:
            avg_shortest_path = nx.average_shortest_path_length(G, weight='weight')
            diameter = nx.diameter(G, weight='weight')  # Diameter does not support weighted versions directly
        else:
            avg_shortest_path = nx.average_shortest_path_length(G)
            diameter = nx.diameter(G)
    else:
        avg_shortest_path = None
        diameter = None
        
    characteristics['average_shortest_path_length'] = avg_shortest_path
    characteristics['diameter'] = diameter

    # Centrality Measures
    centrality_stats = compute_centrality_statistics(G, weighted=weighted, connected=characteristics['is_connected'])
    characteristics.update(centrality_stats)

    # Density
    characteristics['density'] = nx.density(G)

    # Assortativity
    characteristics['assortativity'] = nx.degree_assortativity_coefficient(G)
    
    #Connected Components
    connected_components = list(nx.connected_components(G))
    characteristics['connected_components'] = len(connected_components)
    characteristics["connected_nodes"] = sum(len(component) for component in connected_components)

    return characteristics


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
        "partial_eval": index_value,
        "weighted": False
    },
    "dfs": {
        "name": "Depth-first Search",
        "output_prefix": "Connected Components:",
        "output_regex": r"\s*(\[.*?\])", 
        "malformed_ouput_regex": r"\s*(\[\[.*?\]\])",
        "precision_recall": partial_recall_precision,
        "partial_eval": existence_value,
        "weighted": False
    },
    "dijkstra": {
        "name": "Dijkstra's",
        "output_prefix": "Distances:",
        "output_regex": r"\s*(\[.*?\])",
        "malformed_ouput_regex": r"\s*(\[[\s\S]*?\])",
        "precision_recall": partial_recall_precision,
        "partial_eval": existence_value,
        "weighted": True
    },
    "floyd_warshall": {
        "name": "Floyd-Warshall",
        "output_prefix": "Distances:",
        "output_regex": r"\s*(\[.*?\])",
        "malformed_ouput_regex": r"\s*(\[[\s\S]*?\])",
        "precision_recall": partial_recall_precision,
        "partial_eval": existence_value,
        "weighted": True
    },
    "mst_prim": {
        "name": "Prim MST",
        "output_prefix": "MST Edges:",
        "output_regex": r"\s*(\[.*?\])",
        "malformed_ouput_regex": r"\s*(\[[\s\S]*?\])",
        "precision_recall": partial_recall_precision,
        "partial_eval": existence_value,
        "weighted": True
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

def write_dill(outfile, data):
    dill.dump(data, open(outfile, 'wb'))

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

def parse_string(input_str):
    return ast.literal_eval(input_str)

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

def unpack_literal(literal_str, gold=False, clean_removal=False):
    """
    Parses a string containing a literal list of tuples and automatically unpacks the tuples.

    Parameters:
    - literal_str: A string containing a Python literal expression, specifically a list of tuples.

    Returns:
    - A list where each tuple from the original list is unpacked, 
      according to specified logic.
    """

    if clean_removal:
        # literal_str = literal_str.replace('\u221e', '\'inf\'').replace('\u221E', '\'inf\'').replace("float('inf')", '\'inf\'').replace("inf", '\'inf\'').replace("INF", '\'inf\'')
        valid_tuple_pattern = re.compile(r'\(\d+, \d+, (?:\d+\.\d+|\d+)\)')
        literal_str = "[" + ", ".join(valid_tuple_pattern.findall(literal_str)) + "]"
    # Parse the string to a Python literal (safely with ast.literal_eval)
    try:
        parsed_literal = parse_string(literal_str)#ast.literal_eval(literal_str)
    except:
        return None
    # Initialize an empty list to store unpacked results
    unpacked_list = []
    
    # Iterate through the parsed list, unpacking each tuple
    for item in parsed_literal:
        if gold and isinstance(item,list):
            item = tuple(item)
        if clean_removal:
            if "inf" in str(item):
                continue
            elif isinstance(item,tuple) and item[0] == item[1]:
                continue
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
    matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
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