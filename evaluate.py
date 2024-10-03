import os
import argparse
import train_utils
from train_utils import IncorrectFormatError
import re
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
import json
from scipy.stats import pointbiserialr
import pandas as pd


os.environ["TOKENIZERS_PARALLELISM"] = "false"

def resolve_traj_links(algorithm, trajectory_links, additional_trajectory_links):
    trajectory_dict = {}
    trajectory_links.extend(additional_trajectory_links)
    
    for link in trajectory_links:
        gold_prompt = link["messages"][0]["content"]
        if algorithm == "mst_prim" and not "chat_" in trajectory_links and not "vanilla" in trajectory_links:
            gold_prompt = link["messages"][0]["content"].replace("; Output Format: MST Edges: [(node1, node2, weight), ...]", "")
        trajectory_dict[gold_prompt] = link["traj_id"]
    
    return trajectory_dict

def evaluate(output_json, algorithm, eval_trajectory_links, test_trajectory_links, graph_analysis=False):
    output = train_utils.load_json(output_json)
    
    if not "no_chat" in eval_trajectory_links:
        int_steps_linked = True
        int_steps_exist = True
        
        try:
            trajectory_dict = resolve_traj_links(algorithm, train_utils.load_json(eval_trajectory_links), train_utils.load_json(test_trajectory_links))
        except FileNotFoundError:
            print("here")
            int_steps_linked = False
            int_steps_exist = False
    else:
        int_steps_linked = False
        int_steps_exist = False
        
    gold_output_pattern = train_utils.ACCEPTED_ALGORITHMS[algorithm]["output_prefix"] + train_utils.ACCEPTED_ALGORITHMS[algorithm]["output_regex"]
    semi_gold_output_pattern = train_utils.ACCEPTED_ALGORITHMS[algorithm]["output_regex"]  
    
    if algorithm in ["dfs", "floyd_warshall", "dijkstra", "mst_prim"]:
        alt_gold_output_pattern = train_utils.ACCEPTED_ALGORITHMS[algorithm]["output_prefix"] + train_utils.ACCEPTED_ALGORITHMS[algorithm]["malformed_ouput_regex"]
        alt_semi_gold_output_pattern = train_utils.ACCEPTED_ALGORITHMS[algorithm]["malformed_ouput_regex"]  
    
    int_results = {}
    graph_characteristics = {}
    graph_characteristic_updates = {}
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
        
        # int_steps_linked = int_steps_exist
       
        error_flag = False
        
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
        
        if not "[" in gold_output:
            continue
        
        
        if "chat_gpt" in output_json and not isinstance(result_dict["message"], str): 
            gold_prompt = result_dict["message"][0]
        elif isinstance(result_dict["message"], list): 
            gold_prompt = result_dict["message"][0]["content"]
        else:
            gold_prompt = result_dict["message"]
        
        if algorithm == "mst_prim":
            gold_prompt = gold_prompt.replace("mst_prim", "Prim MST").replace("; Output Format: MST Edges: [(node1, node2, weight), ...]", "")
        
        if int_steps_exist: 
            try:
                traj_id = result_dict["traj_id"]
            except KeyError:
                try:
                    traj_id = trajectory_dict[gold_prompt]
                except KeyError:
                    trajectory_dict[gold_prompt] = -i
                    traj_id = trajectory_dict[gold_prompt]
        else:
            traj_id = i
           
            
        gold_extracted = train_utils.find_last_match(gold_output_pattern, gold_output)        
            
        partial_gold_extracted = train_utils.find_last_match(semi_gold_output_pattern, gold_output)
        
        pred_extracted = train_utils.find_last_match(gold_output_pattern, pred_output)

        partial_pred_extracted = train_utils.find_last_match(semi_gold_output_pattern, pred_output)

        if pred_extracted == None and algorithm in ["floyd_warshall"]:
            pred_extracted = train_utils.find_last_match(alt_gold_output_pattern, pred_output)
        
        if pred_extracted != None:
            pred = train_utils.unpack_literal(pred_extracted) if not algorithm in ["floyd_warshall", "dijkstra", "mst_prim"] else train_utils.unpack_literal(pred_extracted, clean_removal=True)
            if algorithm == "dfs" and pred == None:
                pred_extracted = train_utils.find_last_match(alt_gold_output_pattern, pred_output)
                pred = train_utils.unpack_literal(pred_extracted, gold=True)
            if pred != None and not error_flag:
                invalid_nodes = train_utils.count_invalid_items(pred)
                error_analysis["invalid_nodes"] += invalid_nodes
                error_flag = True if invalid_nodes > 0 else False
        else:
            if not error_flag:
                error_analysis["missing_prefix"] += 1
                error_flag = True
            pred = None 
       
        gold = train_utils.unpack_literal(gold_extracted, gold=True) 
        
        if algorithm == "dfs" and gold == None:
            gold_extracted = train_utils.find_last_match(alt_gold_output_pattern, gold_output)
            gold = train_utils.unpack_literal(gold_extracted, gold=True)
        
        if partial_pred_extracted == None and algorithm in ["floyd_warshall"]:
            partial_pred_extracted = train_utils.find_last_match(alt_semi_gold_output_pattern, pred_output)
            
        if partial_pred_extracted != None:
            partial_pred = train_utils.unpack_literal(partial_pred_extracted) if not algorithm in ["floyd_warshall", "dijkstra", "mst_prim"] else train_utils.unpack_literal(partial_pred_extracted, clean_removal=True)
            if algorithm == "dfs" and partial_pred == None:
                partial_pred_extracted = train_utils.find_last_match(alt_semi_gold_output_pattern, pred_output)
                partial_pred = train_utils.unpack_literal(partial_pred_extracted, gold=True)
            if pred == None:
                if partial_pred != None and not error_flag:
                    error_analysis["invalid_nodes"] += train_utils.count_invalid_items(partial_pred)
                    error_flag = True
        else:
            partial_pred = None
            
        partial_gold = train_utils.unpack_literal(partial_gold_extracted, gold=True)
        if algorithm == "dfs" and partial_gold == None:
            partial_gold_extracted = train_utils.find_last_match(alt_semi_gold_output_pattern, gold_output)
            partial_gold = train_utils.unpack_literal(partial_gold_extracted, gold=True)
        
        if pred != None:
            pred_precision, pred_recall, pred_f1, error_dict = train_utils.ACCEPTED_ALGORITHMS[algorithm]["precision_recall"](pred, gold) 
            if not error_flag:
                error_analysis["hallucinations"] += len(error_dict["fp"])
                error_analysis["missing_nodes"] += len(error_dict["fn"])
                # error_analysis["out_of_order"] += len(error_dict["ooo"])
                error_flag = True
        else:
            pred_precision, pred_recall, pred_f1 = 0, 0, 0
       
        if partial_pred != None:
            partial_pred_precision, partial_pred_recall, partial_pred_f1, partial_error_dict = train_utils.ACCEPTED_ALGORITHMS[algorithm]["precision_recall"](partial_pred, gold) 
            if pred == None and not error_flag:
                error_analysis["hallucinations"] += len(partial_error_dict["fp"])
                error_analysis["missing_nodes"] += len(partial_error_dict["fn"])
                # error_analysis["out_of_order"] += len(partial_error_dict["ooo"])
                error_flag = True
        else:
            partial_pred_precision, partial_pred_recall, partial_pred_f1 = 0, 0, 0
            
        try:
            accuracy.append(int(sorted(pred) == sorted(gold)))
        except:
            accuracy.append(int(pred == gold))
        
        avg_f1.append(pred_f1)
        
        try:
            partial_accuracy.append(int(sorted(partial_pred) == sorted(partial_gold)))
        except:
            partial_accuracy.append(int(partial_pred == partial_gold))
            
        avg_partial_f1.append(partial_pred_f1)
        
        if int_steps_exist:
            if not traj_id in int_results:
                int_results[traj_id] = {
                    "accuracy": [],
                    "f1": [],
                    "partial_accuracy": [],
                    "partial_f1": []
                }
            int_results[traj_id]["accuracy"].append(gold == pred)

            
        results.append({
            "gold_output": gold_output,
            "pred_output": pred_output,
            "accuracy (Exact Match)": gold == pred,
            "f1 (Exact Match)": pred_f1,
            "accuracy (List Only)": partial_pred == partial_gold,
            "f1 (List Only)": partial_pred_f1,
            })
        
        error_analysis["Total_Examples"] += 1
        
        if graph_analysis:
            edgelist, source_node = train_utils.extract_graph_information(gold_prompt, semi_gold_output_pattern)
            graph_characteristic_updates[traj_id] = train_utils.compute_graph_characteristics(edgelist, weighted=train_utils.ACCEPTED_ALGORITHMS[algorithm]["weighted"], source_node=source_node)
    
    int_metrics_dict = {
        "accuracy_score": -1.0,
        "int_accuracy_score": -1.0,
        "final_step_accuracy": -1.0,
        "accuracy_final_step_correlation": -1.0,
        "p_value": -1.0,
        "avg_f1_score": -1.0,
        "partial_accuracy_score": -1.0,
        "partial_avg_f1_score": -1.0,
    }
    
    trajectory_lengths = []
    int_distribution = {}
    
    if int_steps_exist:
        fs_acc = []
        int_steps_acc = []
        
        for traj_id in int_results:
            
            if len(int_results[traj_id]['accuracy']) > 1:
                int_steps_acc.append(np.mean(int_results[traj_id]['accuracy'][:-2]))
            
                fs_acc.append(int_results[traj_id]['accuracy'][-2])#(int_results[traj_id]['final_step_accuracy'])
            else:
                fs_acc.append(int_results[traj_id]['accuracy'][-1])
            trajectory_lengths.append(len(int_results[traj_id]['accuracy']))   
            
            for i, step in enumerate(int_results[traj_id]['accuracy'][:-1]):
                if not f"step_{i}" in int_distribution:
                    int_distribution[f"step_{i}"] = []
                int_distribution[f"step_{i}"].append(step)
                
        try:
            correlation, p_val = tuple(pointbiserialr(fs_acc, int_steps_acc))
        except:
            correlation = -10
            p_val = -10

        mean_accuracy_score = [np.mean(int_results[traj_id]['accuracy']) for traj_id in int_results]
        
        int_metrics_dict = {
            "accuracy_score": np.mean(mean_accuracy_score),
            "int_accuracy_score": np.mean(int_steps_acc), 
            "final_step_accuracy": np.mean(fs_acc),#np.mean([int_results[traj_id]['final_step_accuracy'] for traj_id in int_results]),
            "accuracy_final_step_correlation": correlation,
            "average_trajectory_length": np.mean(trajectory_lengths),
            "p_value": p_val,
            "avg_f1_score": np.mean([np.mean(int_results[traj_id]['f1']) for traj_id in int_results]),
            "partial_accuracy_score": np.mean([np.mean(int_results[traj_id]['partial_accuracy']) for traj_id in int_results]),
            "partial_avg_f1_score": np.mean([np.mean(int_results[traj_id]['partial_f1']) for traj_id in int_results]),
        }

        if graph_analysis:
            graph_characteristics["traj_accuracy_score"] = mean_accuracy_score
            graph_characteristics["final_step_accuracy"] = fs_acc
            graph_characteristics["int_accuracy_score"] = int_steps_acc
            graph_characteristics["trajectory_lengths"] = trajectory_lengths
    if graph_analysis:
        graph_characteristics["accuracy_score"] = accuracy
        for traj_id in graph_characteristic_updates:
            for characteristic, characteristic_value in graph_characteristic_updates[traj_id].items():
                # correlation, p_val = pointbiserialr(characteristic_value, int_steps_acc)
                if not characteristic in graph_characteristics:
                    graph_characteristics[characteristic] = []
                graph_characteristics[characteristic].append(characteristic_value)
    
    metrics_dict = {
        "accuracy_score": sum(accuracy)/len(accuracy) if len(accuracy) > 0 else 0.0,
        "avg_f1_score": sum(avg_f1)/len(avg_f1) if len(avg_f1) > 0 else 0.0,
        "partial_accuracy_score": sum(partial_accuracy)/len(partial_accuracy) if len(partial_accuracy) > 0 else 0.0,
        "partial_avg_f1_score": sum(avg_partial_f1)/len(avg_partial_f1) if len(avg_partial_f1) > 0 else 0.0,
    }

    results.insert(0, metrics_dict)
    
    return metrics_dict, int_metrics_dict, results, graph_characteristics, error_analysis, int_distribution

def main(base_paths, graph_sizes):
    aggregated_results = {}
    aggregated_errors = {}
    aggregated_int_results = {}
    aggregated_int_steps_distribution = {}
    aggregated_graph_characteristics = {}
    results_filter = {}
    graph_analysis = False
    
    for base_path in base_paths:
        # if "local2" in base_path:
        #     continue
        
        for algorithm in os.listdir(base_path):
            if algorithm in train_utils.ACCEPTED_ALGORITHMS:
                
                for graph_size in graph_sizes:
                    model_output_dirs = os.path.join(base_path, f"{algorithm}/graph_size_{graph_size}/llm_data/")
                    for model_type in os.listdir(model_output_dirs):
                        if model_type == "chat":
                            model_name, model_id = 'Llama-Instruct', "Llama"
                        elif model_type == "vanilla":
                            model_name, model_id = "Llama", "Llama"
                        elif model_type == "vanilla_mistral":
                            model_name, model_id = "Mistral", "Mistral"
                        elif model_type == "chat_mistral":
                            model_name, model_id = 'Mistral-Instruct', "Mistral"
                        elif model_type == "chat_gpt":
                            model_name, model_id = 'GPT-4o', 'gpt'
                        elif "rerun" in model_type:
                            continue
                            model_name, model_id = "Llama-Instruct", "Llama"# + "".join(model_type.split("_")[-2:])
                        else:
                            print(f"{model_type} is not a valid model directory name.")
                        reasoning_strategies_dir = os.path.join(model_output_dirs, model_type)
                        for reasoning_strategy in os.listdir(reasoning_strategies_dir):
                            if not model_name in ["Llama-Instruct", 'GPT-4o'] or not reasoning_strategy in ["IO_w_IS", "Int_Steps"]:
                                continue
                            
                            if model_name == 'GPT-4o' and "local2" in reasoning_strategies_dir:
                                continue
                            
                            if algorithm != "dfs":
                                continue
                            
                            model_dir = os.path.join(reasoning_strategies_dir, reasoning_strategy)
                            if "Wndw" in model_dir: #or ("local2" in reasoning_strategies_dir and reasoning_strategy == "Int_Steps" and "gpt" == model_id):
                                continue
                          
                            for file_ in os.listdir(model_dir):
                                if (model_id in file_ or model_name.split("_")[0] in file_) and "_inference" in file_.lower():
                                    try:
                                        metrics_dict, int_metrics_dict, results, graph_characteristics, error_analysis, int_distribution = evaluate(os.path.join(model_dir, file_), algorithm, os.path.join(model_dir.replace("local2", "local"), "evaluation.json"), os.path.join(model_dir.replace("local2", "local"), "test.json"), graph_analysis=graph_analysis)
                                    except json.decoder.JSONDecodeError:
                                        print(f"Algorithm: {algorithm}, Graph Size: {graph_size}, Model: {model_name}, Reasoning Strategy: {reasoning_strategy} has an incorrect JSON format")
                                        continue

                                    # reasoning_strategy = reasoning_strategy if not '_no_chat' in reasoning_strategy else 'IO'
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
                                        
                                        aggregated_int_results[run_name] = {
                                            "Algorithm": algorithm,
                                            "Graph Size": graph_size,
                                            "Model Name": model_name,
                                            "Reasoning Strategy": reasoning_strategy,
                                            **int_metrics_dict
                                        }
                                        
                                        aggregated_errors[run_name] = {
                                            "Algorithm": algorithm,
                                            "Graph Size": graph_size,
                                            "Model Name": model_name,
                                            "Reasoning Strategy": reasoning_strategy,
                                            **error_analysis
                                        } 
                                        
                                        aggregated_int_steps_distribution[run_name] = {
                                            "Algorithm": algorithm,
                                            "Graph Size": graph_size,
                                            "Model Name": model_name,
                                            "Reasoning Strategy": reasoning_strategy,
                                            "Int. Steps Distribution": int_distribution
                                        }
                                        
                                        if graph_analysis:
                                            agg_runs_name = f"{algorithm}_{model_name}_{reasoning_strategy}"
                                            
                                            if "traj_accuracy_score" in graph_characteristics:
                                                multiplier = len(graph_characteristics["traj_accuracy_score"]) 
                                            elif not "IO_no_chat" in reasoning_strategy:
                                                multiplier = 0
                                            else:
                                                multiplier = len(graph_characteristics["accuracy_score"])
                                            
                                            if not agg_runs_name in aggregated_graph_characteristics:
                                                aggregated_graph_characteristics[agg_runs_name] = {
                                                    "algorithm": algorithm,
                                                    "model_name": model_name,
                                                    "reasoning_strategy": reasoning_strategy,
                                                    "graph_size": [graph_size] * multiplier,
                                                    **graph_characteristics
                                                }
                                            else:
                                                aggregated_graph_characteristics[agg_runs_name]["graph_size"].extend([graph_size] * multiplier)
                                                for characteristic in aggregated_graph_characteristics[agg_runs_name]:
                                                    if characteristic in ["graph_size", "model_name", "reasoning_strategy", "algorithm"]:
                                                        continue
                                                    aggregated_graph_characteristics[agg_runs_name][characteristic].extend(graph_characteristics[characteristic])
                                                                                                     
    evaluation_output_dir = os.path.join("/local/ataylor2/algorithmic_reasoning/evaluation")
    os.makedirs(evaluation_output_dir, exist_ok=True)
    
    train_utils.save_results_to_csv(aggregated_results.values(), os.path.join(evaluation_output_dir, 'aggregated_results.csv'))
    train_utils.save_results_to_excel(aggregated_results.values(), os.path.join(evaluation_output_dir, 'aggregated_results.xlsx'))
    train_utils.save_results_to_latex(aggregated_results.values(), os.path.join(evaluation_output_dir, 'aggregated_results.tex'))  
    
    train_utils.save_results_to_csv(aggregated_int_results.values(), os.path.join(evaluation_output_dir, 'aggregated_int_results.csv'))
    train_utils.save_results_to_excel(aggregated_int_results.values(), os.path.join(evaluation_output_dir, 'aggregated_int_results.xlsx'))
    train_utils.save_results_to_latex(aggregated_int_results.values(), os.path.join(evaluation_output_dir, 'aggregated_int_results.tex'))  
    
    # df_graph_characteristics = pd.DataFrame(aggregated_graph_characteristics)
    # df_graph_characteristics = df_graph_characteristics.replace({np.nan: None})
    # df_graph_characteristics.to_json(os.path.join(evaluation_output_dir, 'aggregated_graph_characteristics.json'), orient='records')
    train_utils.write_dill(os.path.join(evaluation_output_dir, 'aggregated_graph_characteristics.pkl'), aggregated_graph_characteristics)
    train_utils.write_dill(os.path.join(evaluation_output_dir, 'aggregated_int_steps_distribution.pkl'), aggregated_int_steps_distribution)
    # train_utils.save_results_to_csv(aggregated_graph_characteristics.values(), os.path.join(evaluation_output_dir, 'aggregated_graph_characteristics.csv'))
    # train_utils.save_results_to_excel(aggregated_graph_characteristics.values(), os.path.join(evaluation_output_dir, 'aggregated_graph_characteristics.xlsx'))
    # train_utils.save_results_to_latex(aggregated_graph_characteristics.values(), os.path.join(evaluation_output_dir, 'aggregated_graph_characteristics.tex'))
    
    train_utils.save_results_to_csv(aggregated_errors.values(), os.path.join(evaluation_output_dir, 'aggregated_errors.csv'))
    train_utils.save_results_to_excel(aggregated_errors.values(), os.path.join(evaluation_output_dir, 'aggregated_errors.xlsx'))
    train_utils.save_results_to_latex(aggregated_errors.values(), os.path.join(evaluation_output_dir, 'aggregated_errors.tex'))                             
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read configuration JSON file')
    parser.add_argument("--base_paths", default=["/local2/ataylor2/algorithmic_reasoning", "/local/ataylor2/algorithmic_reasoning"], type=list)
    parser.add_argument("-graph_sizes", "--graph_sizes", type=list, default=[5,6,7,8,9,10,11,12,13,14,15,20,50], help="Number of nodes present in the graphs generated. Default behavior sets num_samples to the number of training datapoints.")
    args = parser.parse_args()
    
    main(args.base_paths, args.graph_sizes)