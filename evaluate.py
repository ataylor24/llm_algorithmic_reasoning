import os
import argparse
import train_utils
from train_utils import IncorrectFormatError
import re
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def evaluate(output_json, algorithm, graph_size):
    output = train_utils.load_json(output_json)
    
    gold_output_pattern = train_utils.ACCEPTED_ALGORITHMS[algorithm]["output_prefix"] + train_utils.ACCEPTED_ALGORITHMS[algorithm]["output_regex"]
    semi_gold_output_pattern = train_utils.ACCEPTED_ALGORITHMS[algorithm]["output_regex"]  
    
    if algorithm == "dfs" and graph_size != 5:
        alt_gold_output_pattern = train_utils.ACCEPTED_ALGORITHMS[algorithm]["output_prefix"] + train_utils.ACCEPTED_ALGORITHMS[algorithm]["malformed_ouput_regex"]
        alt_semi_gold_output_pattern = train_utils.ACCEPTED_ALGORITHMS[algorithm]["malformed_ouput_regex"]  
    
    results = []
    accuracy = []
    partial_accuracy = []
    avg_f1 = []
    avg_partial_f1 = []
    
    error_analysis = {
        "Total_Examples": 0,
        "missing_prefix": 0,
        "missing_nodes": 0,
        "hallucinations": 0,
        "invalid_nodes": 0,
        "invalid_format": 0,
        "out_of_order": 0
    }
 
    for i, result_dict in enumerate(output):
        if isinstance(result_dict, str) or isinstance(result_dict, list):
            continue
        
        try:
            gold_output = result_dict["ground_truth"]["content"]
        except TypeError:
            gold_output = result_dict["ground_truth"]
                    
        if isinstance(result_dict["pred"], list) and len(result_dict["pred"]) == 0:
            pred_output = ""
        else:
            try:
                pred_output = result_dict["pred"]["generated_text"]
            except TypeError:
                pred_output = result_dict["pred"][0]
        
        gold_extracted = train_utils.find_last_match(gold_output_pattern if algorithm != "dfs" else alt_gold_output_pattern, gold_output)
        pred_extracted = train_utils.find_last_match(gold_output_pattern, pred_output)
        
        partial_gold_extracted = train_utils.find_last_match(semi_gold_output_pattern if algorithm != "dfs" else alt_semi_gold_output_pattern, gold_output)
        partial_pred_extracted = train_utils.find_last_match(semi_gold_output_pattern, pred_output)
        
        if pred_extracted != None:
            pred = train_utils.unpack_literal(pred_extracted.group(1)) 
            if pred != None:
                error_analysis["invalid_nodes"] += train_utils.count_invalid_items(pred)
            else:
                error_analysis["invalid_format"] += 1
        else:
            error_analysis["missing_prefix"] += 1
            pred = None
            
        gold = train_utils.unpack_literal(gold_extracted.group(1), gold=True) 
        
        if partial_pred_extracted != None:
            partial_pred = train_utils.unpack_literal(partial_pred_extracted.group(1)) 
            if pred == None:
                if partial_pred != None:
                    error_analysis["invalid_nodes"] += train_utils.count_invalid_items(partial_pred)
                else:
                    error_analysis["invalid_format"] += 1
        else:
            partial_pred = None
            
        partial_gold = train_utils.unpack_literal(partial_gold_extracted.group(1), gold=True)
        
        if pred != None:
            pred_precision, pred_recall, pred_f1, error_dict = train_utils.ACCEPTED_ALGORITHMS[algorithm]["precision_recall"](pred, gold) 
            error_analysis["hallucinations"] += len(error_dict["fp"])
            error_analysis["missing_nodes"] += len(error_dict["fn"])
            error_analysis["out_of_order"] += len(error_dict["ooo"])
        else:
            pred_precision, pred_recall, pred_f1 = 0, 0, 0
       
        
        if partial_pred != None:
            partial_pred_precision, partial_pred_recall, partial_pred_f1, partial_error_dict = train_utils.ACCEPTED_ALGORITHMS[algorithm]["precision_recall"](partial_pred, gold) 
            if pred == None:
                error_analysis["hallucinations"] += len(partial_error_dict["fp"])
                error_analysis["missing_nodes"] += len(partial_error_dict["fn"])
                error_analysis["out_of_order"] += len(partial_error_dict["ooo"])
        else:
            partial_pred_precision, partial_pred_recall, partial_pred_f1 = 0, 0, 0
            
        # gold_f1 = train_utils.ACCEPTED_ALGORITHMS[algorithm]["indicator_func"](gold, gold, graph_size) #train_utils.list_to_indicator(algorithm, gold, gold)
        # partial_gold_f1 = train_utils.ACCEPTED_ALGORITHMS[algorithm]["indicator_func"](partial_gold, gold, graph_size) #train_utils.list_to_indicator(algorithm, partial_gold, gold)
        
        # accuracy.append(int(pred == gold))
        accuracy.append(int(pred == gold))
        avg_f1.append(pred_f1)
        
        # partial_accuracy.append(int(partial_pred == partial_gold))
        try:
            partial_accuracy.append(int(sorted(partial_pred) == sorted(partial_gold)))
        except:
            partial_accuracy.append(int(partial_pred == partial_gold))
            
        avg_partial_f1.append(partial_pred_f1)

        results.append({
            "gold_output": gold_output,
            "pred_output": pred_output,
            "accuracy (Exact Match)": gold == pred,
            "f1 (Exact Match)": pred_f1,
            "accuracy (List Only)": partial_pred == partial_gold,
            "f1 (List Only)": partial_pred_f1,
            })
        error_analysis["Total_Examples"] += 1
    
    
    metrics_dict = {
        "accuracy_score": sum(accuracy)/len(accuracy) if len(accuracy) > 0 else 0.0,
        "avg_f1_score": sum(avg_f1)/len(avg_f1) if len(avg_f1) > 0 else 0.0,
        "partial_accuracy_score": sum(partial_accuracy)/len(partial_accuracy) if len(partial_accuracy) > 0 else 0.0,
        "partial_avg_f1_score": sum(avg_partial_f1)/len(avg_partial_f1) if len(avg_partial_f1) > 0 else 0.0,
    }

    results.insert(0, metrics_dict)
    
    return metrics_dict, results, error_analysis

def main(base_paths, graph_sizes):
    aggregated_results = {}
    aggregated_errors = {}
    
    results_filter = {}
    for base_path in base_paths:
        for algorithm in os.listdir(base_path):
            if algorithm in train_utils.ACCEPTED_ALGORITHMS:
                for graph_size in graph_sizes:
                    model_output_dirs = os.path.join(base_path, f"{algorithm}/graph_size_{graph_size}/llm_data/")
                    for model_type in os.listdir(model_output_dirs):
                        if not "_" in model_type:
                            model_name = 'Llama'
                        elif "mistral" in model_type:
                            model_name = 'Mistral'
                        elif "gpt" in model_type:
                            model_name = 'gpt'
                        elif "rerun" in model_type:
                            model_name = "Llama"# + "".join(model_type.split("_")[-2:])
                        else:
                            raise NotImplementedError("Model has not been implemented yet.")
                        reasoning_strategies_dir = os.path.join(model_output_dirs, model_type)
                        for reasoning_strategy in os.listdir(reasoning_strategies_dir):

                            model_dir = os.path.join(reasoning_strategies_dir, reasoning_strategy)
                            for file_ in os.listdir(model_dir):
                                if (model_name in file_ or model_name.split("_")[0] in file_) and "_inference" in file_.lower():
                                  
                                    try:
                                        metrics_dict, results, error_analysis = evaluate(os.path.join(model_dir, file_), algorithm, graph_size)
                                    except json.decoder.JSONDecodeError:
                                        print(f"Algorithm: {algorithm}, Graph Size: {graph_size}, Model: {model_name}, Reasoning Strategy: {reasoning_strategy} has an incorrect JSON format")
                                    except:
                                        print(os.path.join(model_dir, file_))
                                    
                                    reasoning_strategy = reasoning_strategy if not '_no_chat' in reasoning_strategy else 'IO'
                                    run_name = f"{algorithm}_{model_name}_{reasoning_strategy}_{graph_size}"

                                    if not run_name in results_filter or train_utils.compare_results(results_filter[run_name], metrics_dict):
                                        results_filter[run_name] = metrics_dict
                                    
                                        aggregated_results[run_name] = {
                                            "Algorithm": algorithm,
                                            "Graph Size": graph_size,
                                            "Model Name": model_name,
                                            "Reasoning Strategy": reasoning_strategy,
                                            **metrics_dict
                                        }
                                        aggregated_errors[run_name] = {
                                            "Algorithm": algorithm,
                                            "Graph Size": graph_size,
                                            "Model Name": model_name,
                                            "Reasoning Strategy": reasoning_strategy,
                                            **error_analysis
                                        }
                                    
                    # str_builder = ""
                    # print(f"Algorithm: {algorithm}, Graph Size: {graph_size}, Model: {model_name}")
                    # for metric, score in metrics_dict.items():
                    #     print(f"{metric}: {score}")
                    #     if "score" in metric:
                    #         str_builder = str_builder + str(score) + ","
                    
                    # print("c/p string:", str_builder)
    
    # evaluation_output_dir = os.path.join(base_path, "evaluation")
    evaluation_output_dir = os.path.join("/local/ataylor2/algorithmic_reasoning/evaluation")
    os.makedirs(evaluation_output_dir, exist_ok=True)
    
    train_utils.save_results_to_csv(aggregated_results.values(), os.path.join(evaluation_output_dir, 'aggregated_results.csv'))
    train_utils.save_results_to_excel(aggregated_results.values(), os.path.join(evaluation_output_dir, 'aggregated_results.xlsx'))
    train_utils.save_results_to_latex(aggregated_results.values(), os.path.join(evaluation_output_dir, 'aggregated_results.tex'))  
    
    train_utils.save_results_to_csv(aggregated_errors.values(), os.path.join(evaluation_output_dir, 'aggregated_errors.csv'))
    train_utils.save_results_to_excel(aggregated_errors.values(), os.path.join(evaluation_output_dir, 'aggregated_errors.xlsx'))
    train_utils.save_results_to_latex(aggregated_errors.values(), os.path.join(evaluation_output_dir, 'aggregated_errors.tex'))                             
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read configuration JSON file')
    parser.add_argument("--base_paths", default=["/local2/ataylor2/algorithmic_reasoning", "/local/ataylor2/algorithmic_reasoning"], type=list)
    parser.add_argument("-graph_sizes", "--graph_sizes", type=list, default=[5,6,7,8,9,10,11,12,13,14,15,20,50], help="Number of nodes present in the graphs generated. Default behavior sets num_samples to the number of training datapoints.")
    args = parser.parse_args()
    
    main(args.base_paths, args.graph_sizes)