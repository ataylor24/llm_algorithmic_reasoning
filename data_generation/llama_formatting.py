import data_utils
import os
import dill

def write_data(output_path, hint_level, dataset_sect, data):
    outfile = os.path.join(output_path, dataset_sect + "_" + hint_level + ".pkl")
    dill.dump(data, open(outfile, 'wb'))
    
def resolve_output_dirs(data_path):
    out_path = os.path.join(data_path, "llama_data")
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    return out_path

def apply_llama_format(dataset_path, algorithm, hint_level):
    data = data_utils.load_json(dataset_path)
    reformatted_data = ""
    for idx in data:
        edge_list = data[idx]["inputs"][0]
        source_node = data[idx]["inputs"][1]
        input = f"Algorithm: {algorithm}; Edgelist: {','.join([str(tuple(edge)) for edge in edge_list])}; Source Node: {source_node}."
        output = f"{data[idx]['outputs']}"
        if hint_level == "data_with_hints":
            input = input + "\n" + data[idx]["hints"]

        reformatted_data += f"<s>[INST] {input} [/INST] {output} </s>"
        
    return reformatted_data

def main():
    algorithm = "bfs"
    data_path = f"/local2/ataylor2/algorithmic_reasoning/{algorithm}/llm_data"
    out_path = resolve_output_dirs(data_path)
    
    ref_datasets = []
    for dataset_sect in ["training", "validation", "testing"]:
        for hint_level in ["data_with_hints", "data_no_hints"]:
            reformatted_data = apply_llama_format(os.path.join(data_path, dataset_sect + ".json"), algorithm, hint_level)
            write_data(out_path, hint_level, dataset_sect, reformatted_data)
    
if __name__ == "__main__":
    main()