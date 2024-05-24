import os
import argparse
import train_utils
import re
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
import ast

os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

def evaluate_(output_path, graph_size):
    
    results = train_utils.load_json(output_path)
    # INPUT_PATTERN = "\[INST\](.*?)\[/INST\]"
    gold_pattern = r"Reachable nodes:\s*(\[.*?\])"
    semi_gold_pattern = r"\s*(\[.*?\])"
    
    accuracy = []
    avg_f1 = []
    f1 = [[], []]
    
    for (gold_output, pred_output, score) in results:        
        print("gold_output", gold_output)
        print("pred_output", pred_output)
    
        # Extract lists using regex, ignoring case
        pred_extracted = re.search(gold_pattern, pred_output, re.IGNORECASE)
        gold_extracted = re.search(gold_pattern, gold_output, re.IGNORECASE)
        
        pred = unpack_literal(pred_extracted.group(1)) if pred_extracted != None else None
        gold = unpack_literal(gold_extracted.group(1))
        
        if pred != None:
            pred_f1 = list_to_indicator(pred, graph_size)  
            
        else:
            pred_f1 = [-1] * graph_size
            
        gold_f1 = list_to_indicator(gold, graph_size)
        print(pred, gold)
        accuracy.append(int(pred == gold))
        avg_f1.append(f1_score(np.array(gold_f1), np.array(pred_f1), average="micro"))
        f1[0].extend(gold_f1)
        f1[1].extend(pred_f1)
                
        print("exact score", gold == pred)
        print("partial score", accuracy_score(np.array(gold_f1), np.array(pred_f1)), f1_score(np.array(gold_f1), np.array(pred_f1), average="micro"))
        print("-----------------")
        
        # partial_match_pred_scores.extend(partial_match_pred_score)
        # partial_match_gold_scores.extend(partial_match_gold_score)
        
        
    return {
        "accuracy_score": sum(accuracy)/len(accuracy),
        "avg_f1_score": sum(avg_f1)/len(avg_f1),
        "f1_score": f1_score(np.array(f1[0]), np.array(f1[1]), average="micro"),
        "f1_confusion_matrix": confusion_matrix(np.array(f1[0]), np.array(f1[1]))
    }, results

def evaluate(output_path, graph_size):
    import json

    results_json = train_utils.load_json(output_path)
    gold_output_pattern = r"Reachable Nodes:\s*(\[.*?\])"
    semi_gold_output_pattern = r"\s*(\[.*?\])"
    
    results = []
    accuracy = []
    partial_accuracy = []
    avg_f1 = []
    avg_partial_f1 = []
    f1 = [[], []]
    partial_f1 = [[], []]

    for i, result_dict in enumerate(results_json):
        
        if i == 0:
            continue
        
        gold_output = result_dict["gold_output"]
        pred_output = result_dict["pred_output"]
        
        print(gold_output)
        print(pred_output)
        
        gold_extracted = re.search(gold_output_pattern, gold_output, re.IGNORECASE)
        pred_extracted = re.search(gold_output_pattern, pred_output, re.IGNORECASE)
        
        partial_gold_extracted = re.search(semi_gold_output_pattern, gold_output, re.IGNORECASE)
        partial_pred_extracted = re.search(semi_gold_output_pattern, pred_output, re.IGNORECASE)
        
        pred = train_utils.unpack_literal(pred_extracted.group(1)) if pred_extracted != None else None
        gold = train_utils.unpack_literal(gold_extracted.group(1))
        
        partial_pred = train_utils.unpack_literal(partial_pred_extracted.group(1)) if partial_pred_extracted != None else None
        partial_gold = train_utils.unpack_literal(partial_gold_extracted.group(1))
        
        if pred != None:
            pred_f1 = train_utils.list_to_indicator(pred, graph_size)  
            
        else:
            pred_f1 = [2] * graph_size
            
        if partial_pred != None:
            partial_pred_f1 = train_utils.list_to_indicator(partial_pred, graph_size)  
            
        else:
            partial_pred_f1 = [2] * graph_size
            
        gold_f1 = train_utils.list_to_indicator(gold, graph_size)
        partial_gold_f1 = train_utils.list_to_indicator(partial_gold, graph_size)
        
        print(pred_output, gold_output)
        
        accuracy.append(int(pred == gold))
        avg_f1.append(f1_score(np.array(gold_f1), np.array(pred_f1), average="micro"))
        f1[0].extend(gold_f1)
        f1[1].extend(pred_f1)
        
        partial_accuracy.append(int(partial_pred == partial_gold))
        avg_partial_f1.append(f1_score(np.array(partial_gold_f1), np.array(partial_pred_f1), average="micro"))
        partial_f1[0].extend(partial_gold_f1)
        partial_f1[1].extend(partial_pred_f1)

                
        results.append({
            "gold_output": gold_output,
            "pred_output": pred_output,
            "accuracy (Exact Match)": gold == pred,
            "f1 (Exact Match)": f1_score(np.array(gold_f1), np.array(pred_f1), average="micro"),
            "accuracy (List Only)": partial_pred == partial_gold,
            "f1 (List Only)": f1_score(np.array(partial_gold_f1), np.array(partial_pred_f1), average="micro"),
            })
    
    print(partial_f1)
    
    metrics_dict = {
        "accuracy_score": sum(accuracy)/len(accuracy),
        "avg_f1_score": sum(avg_f1)/len(avg_f1),
        "f1_confusion_matrix": confusion_matrix(np.array(f1[0]), np.array(f1[1])).tolist(),
        "partial_accuracy_score": sum(partial_accuracy)/len(partial_accuracy),
        "partial_avg_f1_score": sum(avg_partial_f1)/len(avg_partial_f1),
        "partial_f1_confusion_matrix": confusion_matrix(np.array(partial_f1[0]), np.array(partial_f1[1])).tolist()
    }
    
    results.insert(0, metrics_dict)
    
    return metrics_dict, results

def main(outfile_path, graph_size):
    
    metrics_dict, results = evaluate(outfile_path, graph_size)
    
    str_builder = ""
    for metric, score in metrics_dict.items():
        print(metric)
        print(score)
        if "score" in metric:
            str_builder = str_builder + str(score) + ","
    
    print("c/p string:", str_builder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read configuration JSON file')
    parser.add_argument('outfile_path', type=str, help='Path to the configuration JSON file')
    parser.add_argument('graph_size', type=int, help='Path to the configuration JSON file')
    args = parser.parse_args()
    
    main(args.outfile_path, args.graph_size)