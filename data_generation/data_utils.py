import argparse
import os
import json
import dill
import yaml

OUTPUT_FORMATS = ['llama2', "mistral7b"]
REASONING_STRATEGIES = ["CoT", "IO"]

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

FORMATTED_ALGORITHMS = {
    "bfs": {
        "name": "Breadth-first Search",
        "goal": "reachability",
        "instruction": "List all known reachable nodes ordered lexicographically smallest to largest."
        },
    "dfs": {
        "name": "Depth-first Search",
        "goal": "reachability",
        "instruction": ""
            },
    "dijkstra": {
        "name": "Dijkstra's",
        "goal": "shortest-path",
        "instruction": "What are the distances between each pair of nodes?"
        },
    "bfd": {
        "name": "Bellman-Ford",
        "goal": "shortest path",
        "instruction": ""
        },
    "floyd_warshall": {
        "name": "Floyd-Warshall",
        "goal": "shortest-path",
        "instruction": "What are the shortest path distances between all pairs of nodes?"
    }
}

TRAIN_TEST_SPLIT = {
    3: [4,4],
    4: [42, 22],
    5: [800, 224]
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
    
def write_llama_chat_format(reasoning_strategy, data_sect, data):
    

    llama_data = []
        
    for idx in data:
        llama_data_inst = []
        algorithm = data[idx]["inputs"][0]
        edge_list = data[idx]["inputs"][1]
        source_node = data[idx]["inputs"][2]
        
        init_prompt = None
        
        for i, execution_step in enumerate(data[idx]["hints"]):
            if i == 0:
                role = "system"
            elif i % 2 == 1:
                role = "assistant"
            else:
                role = "user"
                
            hints = execution_step + "\n" if (reasoning_strategy != "IO" and data_sect != "evaluation") or i % 2 == 1 else ""

            if i == 0:
                if source_node == "":
                    content = f"Please perform the {FORMATTED_ALGORITHMS[algorithm]['name']} algorithm for {FORMATTED_ALGORITHMS[algorithm]['goal']} on the following graph: Edgelist: [{','.join([str(tuple(edge)) for edge in edge_list])}]; \n{hints}\n{FORMATTED_ALGORITHMS[algorithm]['instruction']}"
                else:
                    content = f"Please perform the {FORMATTED_ALGORITHMS[algorithm]['name']} algorithm for {FORMATTED_ALGORITHMS[algorithm]['goal']} on the following graph: Edgelist: [{','.join([str(tuple(edge)) for edge in edge_list])}]; Source Node: {str(source_node)}\n{hints}\n{FORMATTED_ALGORITHMS[algorithm]['instruction']}"
                init_prompt = content
            elif i + 1 >= len(data[idx]["hints"]):
                content =  data[idx]["outputs"]
            elif i % 2 == 1:
                content =f"{hints}"
            else:
                content =f"{hints}{FORMATTED_ALGORITHMS[algorithm]['instruction']}"
                
            llama_data_inst.append({
                "role": role,
                "content": content,
            })
                
        # llama_data.append(llama_data_inst) 
        llama_data.append({
                    'prompt': init_prompt, 
                    'chosen': llama_data_inst,
                    'rejected': {} if reasoning_strategy in ["ToT", "GoT"] else None,
                    'messages': llama_data_inst,
                    'score_chosen': 1,
                    'score_rejected': 1
                })

        # outfile = os.path.join(os.path.join(output_dir, reasoning_strategy), data_sect)
        
        # out_yml = os.path.join(os.path.join(output_dir, reasoning_strategy), "README.md")
        
        # write_data_config_readme(out_yml)
        
        # write_json(outfile + ".json", llama_data)
        
    return llama_data

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
    parser.add_argument("algorithm", type=str, choices=['bfs', 'dfs', 'dijkstra', 'floyd_warshall'], 
                        help="Algorithm must be one of: 'bfs', 'dfs', 'dijkstra', or 'floyd_warshall'.")
    parser.add_argument("-training_style", "--training_style", type=str, default="QA",choices=['QA', 'AoT'], #CoT, ToT, GoT
                        help="Algorithm must be one of: 'bfs', 'dfs','dka', or 'bfd'.")
    parser.add_argument("-graph_sizes", "--graph_sizes", type=int, default=10, help="Number of nodes present in the graphs generated. Default behavior sets num_samples to the number of training datapoints.")
    parser.add_argument("-num_samples", "--num_samples", type=int, default=-1, help="Number of data samples to generate.")
    parser.add_argument("-neg_edges", "--neg_edges", type=bool, default=True, help="Include negative edges, ex. '0 is not reachable from 1'.")
    parser.add_argument("-seed", "--seed", type=int, default=100898, help="Random seed used in constructing the CLRS sampler; the default is 10081998.")
    parser.add_argument("-output_dir", "--output_dir", type=str, default="/local2/ataylor2/algorithmic_reasoning", help="Output directory. Will create folders named after the algorithm for which data is generated.")
    parser.add_argument("-train_test_split", "--train_test_split", type=list, default=[1000,500], help="Training/Testing split ratios. The Test set will be equally split into Validation and Test.")
    parser.add_argument("-output_formats", "--output_formats", type=list, default=["llama2", "mistral7b"], choices=OUTPUT_FORMATS, help="Output format for dataset")
    # Parse the arguments
    args = parser.parse_args()

    return args
