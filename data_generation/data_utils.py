import argparse
import os
import json
import dill

OUTPUT_FORMATS = ['llama2']
HINT_LEVELS = ["with_hints", "no_hints"]

FORMATTED_ALGORITHMS = {
    "bfs": {
        "name": "Breadth-first Search",
        "goal": "reachability"
        },
    "dfs": {
        "name": "Depth-first Search",
        "goal": "reachability"
            },
    "dka": {
        "name": "Dijkstra's",
        "goal": "shortest-path"
        },
    "bfd": {
        "name": "Bellman-Ford",
        "goal": "shortest path"
        }
}

TRAIN_TEST_SPLIT = {
    3: [4,4],
    4: [42, 22],
    5: [800, 224]
}

def load_json(filepath):
    return json.load(open(filepath, 'r'))

def write_json(outfile, data):
    json.dump(data, open(outfile, "w"))
        
def write_pickle(outfile, data):
    dill.dump(data, open(os.path.join(outfile), 'wb'))

def write_clrs_format(outfile, data):
    write_pickle(outfile, data)



def write_llama_format_deprecated(output_dir, data_sect, data):
    warnings.warn("write_llama_format_deprecated is deprecated and may be removed in a future version. Use <write_llama_format> instead.", DeprecationWarning, stacklevel=2)
    for hint_level in HINT_LEVELS:
        llama_data = []
        for idx in data:
            llama_data.append(apply_llama_format(data[idx], hint_level, data_sect))
        
        outfile = os.path.join(os.path.join(output_dir,hint_level), data_sect)
        
        write_json(outfile + ".json", llama_data)
        
def write_llama_format(output_dir, data_sect, data):
    for hint_level in HINT_LEVELS:
        llama_data = []
            
        for idx in data:
            
            algorithm = data[idx]["inputs"][0]
            edge_list = data[idx]["inputs"][1]
            source_node = data[idx]["inputs"][2]
                        
            
            llama_data_inst = {
                "instruction": f"Please perform the {FORMATTED_ALGORITHMS[algorithm]['name']} algorithm for {FORMATTED_ALGORITHMS[algorithm]['goal']}.",
                "input": f"Edgelist: [{','.join([str(tuple(edge)) for edge in edge_list])}]; Source Node: {str(source_node)}.",
                "output": data[idx]["outputs"]
            }
            if hint_level != "no_hints" and data_sect != "testing":
                llama_data_inst["instruction"] += "\n### Steps:" + data[idx]["hints"]

            llama_data.append(llama_data_inst)  
        
        outfile = os.path.join(os.path.join(output_dir,hint_level), data_sect)
        
        write_json(outfile + ".json", llama_data)

# def write_llama_format(output_dir, data_sect, data):
#     for hint_level in HINT_LEVELS:
        
#         llama_data = { 
#             "algorithm_name": [],
#             "algorithm_goal": [],
#             "edgelist": [],
#             "source_node": [],
#             "outputs": []
#         }
#         if hint_level != "no_hints" and data_sect != "testing":
#             llama_data["hints"] = []
            
#         for idx in data:
#             algorithm = data[idx]["inputs"][0]
#             edge_list = data[idx]["inputs"][1]
#             source_node = data[idx]["inputs"][2]
            
#             llama_data["algorithm_name"].append(FORMATTED_ALGORITHMS[algorithm]['name'])
#             llama_data["algorithm_goal"].append(FORMATTED_ALGORITHMS[algorithm]['goal'])
#             llama_data["edgelist"].append(','.join([str(tuple(edge)) for edge in edge_list]))
#             llama_data["source_node"].append(source_node)
#             llama_data["outputs"].append(data[idx]["outputs"])
            
#             if hint_level != "no_hints" and data_sect != "testing":
#                 llama_data["hints"].append(data[idx]["hints"])
            
        
#         outfile = os.path.join(os.path.join(output_dir,hint_level), data_sect)
        
#         write_json(outfile + ".json", llama_data)

def apply_llama_format(data_inst, hint_level, dataset_view):
    warnings.warn("apply_llama_format is deprecated and may be removed in a future version. Use <write_llama_format> instead.", DeprecationWarning, stacklevel=2)

    algorithm = data_inst["inputs"][0]
    edge_list = data_inst["inputs"][1]
    source_node = data_inst["inputs"][2]
    
    inputs = f"Edgelist: [{','.join([str(tuple(edge)) for edge in edge_list])}]; Source Node: {source_node}. Please perform the {FORMATTED_ALGORITHMS[algorithm]['name']} algorithm for {FORMATTED_ALGORITHMS[algorithm]['goal']}."
    output = f"{data_inst['outputs']}"
    if not dataset_view == "testing" and hint_level == "with_hints":
        inputs = inputs + "\n" + data_inst["hints"]

    return {"text": f"[INST]\n{inputs}[/INST]\n\n[INST]\n{output}[/INST]"}
    
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
        for hint_level in HINT_LEVELS:
            llm_formatted_with_hint_level_data_dir = os.path.join(llm_formatted_data_dir, hint_level)
            if not os.path.exists(llm_formatted_with_hint_level_data_dir):
                os.mkdir(llm_formatted_with_hint_level_data_dir)
            
            
    
    return clrs_data_dir, llm_formatted_data_dirs

    
def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(description="Jointly constructs CLRS data and sequence-to-sequence data.")

    # Add arguments
    parser.add_argument("algorithm", type=str, choices=['bfs', 'dfs', 'dka', 'bfd'], 
                        help="Algorithm must be one of: 'bfs', 'dfs','dka', or 'bfd'.")
    parser.add_argument("-graph_sizes", "--graph_sizes", type=int, default=3, help="Number of nodes present in the graphs generated. Default behavior sets num_samples to the number of training datapoints.")
    parser.add_argument("-num_samples", "--num_samples", type=int, default=-1, help="Number of data samples to generate.")
    parser.add_argument("-neg_edges", "--neg_edges", type=bool, default=True, help="Include negative edges, ex. '0 is not reachable from 1'.")
    parser.add_argument("-seed", "--seed", type=int, default=100898, help="Random seed used in constructing the CLRS sampler; the default is 10081998.")
    parser.add_argument("-output_dir", "--output_dir", type=str, default="/Users/inmancosta/Documents/GitHub/llm_algorithmic_reasoning", help="Output directory. Will create folders named after the algorithm for which data is generated.")
    parser.add_argument("-train_test_split", "--train_test_split", type=list, default=[1000,500], help="Training/Testing split ratios. The Test set will be equally split into Validation and Test.")
    parser.add_argument("-output_formats", "--output_formats", type=list, default=["llama2"], choices=OUTPUT_FORMATS, help="Output format for dataset")
    # Parse the arguments
    args = parser.parse_args()

    return args