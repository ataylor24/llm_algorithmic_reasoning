import argparse
import os
import json
import dill
import yaml

OUTPUT_FORMATS = ["vanilla", "chat", "vanilla_mistral", "chat_mistral", "chat_gpt"] #["chat_rerun_ralpha_8", "chat_rerun_ralpha_32"]
# OUTPUT_FORMATS = ["vanilla", "vanilla_mistral", "chat_mistral"] #["chat_rerun_ralpha_8", "chat_rerun_ralpha_32"]

# REASONING_STRATEGIES = ["IO_no_chat"]#, "IO", "Int_Steps"]

YML_OUTFILE = {
    "configs": [
        {
            "config_name": "train",
            "data_files": "training.json"
        },
        {
            "config_name": "test",
            "data_files": "testing.json"
        }
    ]
}

REASONING_STRATEGIES = {
    # "IO_w_IS_no_tf": "",
    # "IO_no_tf": "",
    "IO_nc_w_IS": ""
    # "IO_w_IS": "",
    # "Int_Steps_w_IO": "",
    # "IO_no_chat": "; Do not provide intermediate steps",
    # "IO": "",
    # "Int_Steps": "",
    # "Int_Steps_Wndw": ""
}

FORMATTED_ALGORITHMS = {
    "bfs": {
        "name": "Breadth-first Search",
        "goal": "reachability",
        "instruction": "List all known reachable nodes ordered lexicographically smallest to largest.",
        "output_format":"; Output Format: Reachable Nodes: [node1, node2, ...]",
        "eval_output_format": "Reachable Nodes:"
        },
    "dfs": {
        "name": "Depth-first Search",
        "goal": "reachability",
        "instruction": "List all known connected components.",
        "output_format":"; Output Format: Connected Components: [(node1, node2), (node3,node5), ...]",
        "eval_output_format": "Connected Components:"
        },
    "dijkstra": {
        "name": "Dijkstra's",
        "goal": "shortest-path",
        "instruction": "What are the distances between each pair of nodes?",
        "output_format":"; Output Format: Distances: [(node1, node2, weight), ...]",
        "eval_output_format": "Distances:"
        },
    "floyd_warshall": {
        "name": "Floyd-Warshall",
        "goal": "shortest-path",
        "instruction": "What are the shortest path distances between all pairs of nodes?",
        "output_format":"; Output Format: Distances: [(node1, node2, weight), ...]",
        "eval_output_format": "Distances:"
        },
    "mst_prim": {
        "name": "Prim MST",
        "goal": "minimum spanning tree",
        "instruction": "What is the edgelist of the minimum spanning tree?",
        "output_format":"; Output Format: MST Edges: [(node1, node2, weight), ...]",
        "eval_output_format": "MST Edges:"
        }
}

TRAIN_TEST_SPLIT = {
    3: [4,4],
    4: [42, 22],
    5: [800, 100]
}

def dump_yml(outfile, data):
    yaml.dump(data, open(outfile, 'w'), default_flow_style=False)

def load_json(filepath):
    return json.load(open(filepath, 'r'))

def write_json(outfile, data):
    json.dump(data, open(outfile, "w"))
        
def write_pickle(outfile, data):
    dill.dump(data, open(outfile, 'wb'))

def write_clrs_format(outfile, data):
    write_pickle(outfile, data)
    
def write_data_config_readme(outfile):
    dump_yml(outfile, YML_OUTFILE)
    
def json_to_string(data):
    prompt = data.get("prompt", "")
    messages = data.get("messages", [])
    
    output_string = prompt + "\n\n"
    
    for message in messages:
        content = message.get("content", "")
        output_string += content + "\n"
    
    return {"content": output_string.strip()}
    
def write_chat_format(reasoning_strategy, data_sect, data, context_window=-1):
    

    chat_data = []
        
    for idx in data:
        llama_data_inst = []
        algorithm = data[idx]["inputs"][0]
        edge_list = data[idx]["inputs"][1]
        source_node = data[idx]["inputs"][2]
        
        init_prompt = None
        
        for i, execution_step in enumerate(data[idx]["hints"]):
            if reasoning_strategy in ["IO_no_chat", "Int_Steps_w_IO"] and i != 0 and i + 1 < len(data[idx]["hints"]):
                continue
            
            if i == 0:
                role = "system"
            elif i % 2 == 1:
                role = "assistant"
            else:
                role = "user"
            
            hints = execution_step + "\n" if (("IS" in reasoning_strategy or not "IO" in reasoning_strategy) and data_sect != "evaluation") or i % 2 == 1 else ""
  
            if i == 0:
                if source_node == "":
                    content = f"Please perform the {FORMATTED_ALGORITHMS[algorithm]['name']} algorithm for {FORMATTED_ALGORITHMS[algorithm]['goal']} on the following graph: Edgelist: [{','.join([str(tuple(edge)) for edge in edge_list])}]{FORMATTED_ALGORITHMS[algorithm]['output_format']}{REASONING_STRATEGIES[reasoning_strategy]}.\n{hints}\n{FORMATTED_ALGORITHMS[algorithm]['instruction']}"
                    init_prompt = f"Please perform the {FORMATTED_ALGORITHMS[algorithm]['name']} algorithm for {FORMATTED_ALGORITHMS[algorithm]['goal']} on the following graph: Edgelist: [{','.join([str(tuple(edge)) for edge in edge_list])}]"
                else:
                    content = f"Please perform the {FORMATTED_ALGORITHMS[algorithm]['name']} algorithm for {FORMATTED_ALGORITHMS[algorithm]['goal']} on the following graph: Edgelist: [{','.join([str(tuple(edge)) for edge in edge_list])}]; Source Node: {str(source_node)}{FORMATTED_ALGORITHMS[algorithm]['output_format']}{REASONING_STRATEGIES[reasoning_strategy]}.\n{hints}\n{FORMATTED_ALGORITHMS[algorithm]['instruction']}"
                    init_prompt = content
            elif i + 1 >= len(data[idx]["hints"]):
                if algorithm in ["bfs"]:
                    content =f"{hints}"
                else:
                    content =  data[idx]["outputs"]
            elif i % 2 == 1:
                content =f"{hints}"
            else:
                content =f"{hints}{FORMATTED_ALGORITHMS[algorithm]['instruction']}"
                
            llama_data_inst.append({
                "role": role,
                "content": content,
            })
        
        if context_window == -1:
            datapoint = {
                        "traj_id": f"{idx}",
                        'prompt': init_prompt, 
                        'messages': llama_data_inst,
                    }
            # chat_data.append(datapoint if not "mistral" in output_format else json_to_string(datapoint))
            chat_data.append(datapoint)
        else:

            windows = []
                 
            loop_start = - (2* context_window) + 2
            
            for i in range(loop_start, len(llama_data_inst), 2):
                if i + context_window*2 > len(llama_data_inst):
                    break
                start_idx = max(0, i)
                end_idx = i + context_window*2 
                
                if start_idx > 0:
                    init_context_prompt = {
                        "role":"system",
                        "content": init_prompt
                        }
                    
                    window = [init_context_prompt] + llama_data_inst[start_idx:end_idx]
                else:
                    window = llama_data_inst[start_idx:end_idx]

                windows.append((start_idx, end_idx))

                datapoint = {
                        "traj_id": f"{idx}.{i}",
                        'prompt': init_prompt, 
                        'messages': window,
                    }
                
                # chat_data.append(datapoint if not "mistral" in output_format else json_to_string(datapoint))
                chat_data.append(datapoint)
    write_json(f"/local2/ataylor2/algorithmic_reasoning/data_samples/{data_sect}_sample.json", chat_data)
        
    return chat_data

def write_llama_format(output_dir, data_sect, data):
    for hint_level in REASONING_STRATEGIES:
        llama_data = []
        for idx in data:
            algorithm = data[idx]["inputs"][0]
            edge_list = data[idx]["inputs"][1]
            source_node = data[idx]["inputs"][2]

            if source_node == "":
                llama_data_inst = {
                "instruction": f"Please perform the {FORMATTED_ALGORITHMS[algorithm]['name']} algorithm for {FORMATTED_ALGORITHMS[algorithm]['goal']} on the following graph:",
                "input": f"Edgelist: [{','.join([str(tuple(edge)) for edge in edge_list])}];.",
                "output": data[idx]["outputs"]
                }
            else:
                llama_data_inst = {
                    "instruction": f"Please perform the {FORMATTED_ALGORITHMS[algorithm]['name']} algorithm for {FORMATTED_ALGORITHMS[algorithm]['goal']} on the following graph:",
                    "input": f"Edgelist: [{','.join([str(tuple(edge)) for edge in edge_list])}]; Source Node: {str(source_node)}.",
                    "output": data[idx]["outputs"]
                }
            if hint_level != "no_hints" and data_sect != "testing":
                llama_data_inst["hints"] = data[idx]["hints"]

            llama_data.append(llama_data_inst)  

        outfile = os.path.join(os.path.join(output_dir,hint_level), data_sect)
        write_json(outfile + ".json", llama_data)
    
def resolve_output_dirs(output_dir, algorithm, output_formats, graph_size):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    parent_data_dir = os.path.join(output_dir, algorithm)
    if not os.path.exists(parent_data_dir):
        os.mkdir(parent_data_dir)
        
    child_data_dir = os.path.join(parent_data_dir, f"graph_size_{graph_size}")
    if not os.path.exists(child_data_dir):
        os.mkdir(child_data_dir)
        
    clrs_data_dir = os.path.join(child_data_dir, "clrs_data")
    llm_data_dir = os.path.join(child_data_dir, "llm_data")
    
    if not os.path.exists(clrs_data_dir):
        os.mkdir(clrs_data_dir)
    if not os.path.exists(llm_data_dir):
        os.mkdir(llm_data_dir)
    
    llm_formatted_data_dirs = {}
    for output_format in output_formats:
        llm_formatted_data_dir = os.path.join(llm_data_dir, output_format)
        llm_formatted_data_dirs[output_format] = llm_formatted_data_dir
        if not os.path.exists(llm_formatted_data_dir):
            os.mkdir(llm_formatted_data_dir)
        for hint_level in REASONING_STRATEGIES:
            llm_formatted_with_hint_level_data_dir = os.path.join(llm_formatted_data_dir, hint_level)
            if not os.path.exists(llm_formatted_with_hint_level_data_dir):
                os.mkdir(llm_formatted_with_hint_level_data_dir)
            
            
    
    return clrs_data_dir, llm_formatted_data_dirs

    
def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(description="Jointly constructs CLRS data and sequence-to-sequence data.")

    # Add arguments
    parser.add_argument("algorithm", type=str, choices=['bfs', 'dfs', 'dijkstra', 'floyd_warshall', 'mst_prim', 'all'], 
                        help="Algorithm must be one of: 'bfs', 'dfs', 'dijkstra', 'floyd_warshall', or 'mst_prim.")
    parser.add_argument("-graph_sizes", "--graph_sizes", type=list, default=[5,6,7,8,9,10,11,12,13,14,15,20,50], help="Number of nodes present in the graphs generated. Default behavior sets num_samples to the number of training datapoints.")
    parser.add_argument("-num_samples", "--num_samples", type=int, default=-1, help="Number of data samples to generate.")
    parser.add_argument("-neg_edges", "--neg_edges", type=bool, default=True, help="Include negative edges, ex. '0 is not reachable from 1'.")
    parser.add_argument("-seed", "--seed", type=int, default=100898, help="Random seed used in constructing the CLRS sampler; the default is 10081998.")
    parser.add_argument("-output_dir", "--output_dir", type=str, default="/local/ataylor2/algorithmic_reasoning", help="Output directory. Will create folders named after the algorithm for which data is generated.")
    parser.add_argument("-train_test_split", "--train_test_split", type=list, default=[1000,100], help="Training/Testing split ratios. The Test set will be equally split into Validation and Test.")
    parser.add_argument("-output_formats", "--output_formats", type=list, default=["vanilla", "chat"], choices=OUTPUT_FORMATS, help="Output format for dataset")
    parser.add_argument("-window_sizes", "--window_sizes", type=list, default=[1,2,5], help="Window sizes for intermediate steps with shorter context windows")
    # Parse the arguments
    args = parser.parse_args()

    return args
