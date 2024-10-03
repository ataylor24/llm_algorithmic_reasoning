import os
import argparse

def generate_run_scripts(alg, sizes, base_path, reasoning_strategy, chat_type, device_mapping, window=-1):
    outer_directory = os.path.join(base_path, 'alg_run_scripts')
    os.makedirs(outer_directory, exist_ok=True)
    window_tag = f'_{window}' if window > 0 else ''
    directory = os.path.join(outer_directory, f"{alg}_{reasoning_strategy}_{chat_type}{window_tag}_scripts")
    os.makedirs(directory, exist_ok=True)
    reasoning_strategy = reasoning_strategy if window == -1 else reasoning_strategy + f"_{window}"
    print(f"reasoning_strategy: {reasoning_strategy}, window: {window}")
    script_template = """export CUDA_VISIBLE_DEVICES={device_id}
export HF_HOME=/local2/ataylor2/algorithmic_reasoning/cache
export HF_TOKEN=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
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file {base_path}recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 --main_process_port={port} {base_path}scripts/run_sft.py /local/ataylor2/algorithmic_reasoning/{alg}/graph_size_{size}/llm_data/{chat_type}/{reasoning_strategy}/config_qlora.yaml --load_in_4bit=true
"""
    if alg == 'bfs':
        port = 15800
    elif alg == 'dfs':
        port = 25800
    elif alg == 'dijkstra':
        port = 35800
    elif alg == 'floyd_warshall':
        port = 45800
    elif alg == 'mst_prim':
        port = 55800
    
    script_names = {}
    
    for i, size in enumerate(sizes):
        script_content = script_template.format(
            alg=alg, 
            size=size, 
            device_id=device_mapping[size], 
            port=port, 
            reasoning_strategy=reasoning_strategy,
            base_path=base_path,
            chat_type=chat_type
        )
        script_filename = os.path.join(directory, f"run_{alg}_size_{size}.sh")
        script_names[size] = script_filename
        with open(script_filename, 'w') as script_file:
            script_file.write(script_content)

        # Increment the device_id, wrapping around to 0 after 6
        #(device_id + 1) % 7 if i % 3 == 1 else device_id
        # Increment the port
        port += 1
    
    return script_names

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate run scripts for different algorithms and graph sizes.")
    parser.add_argument('--alg', default='dfs', choices=['bfs', 'dfs', 'dijkstra', 'floyd_warshall', 'mst_prim'], help="Algorithm to generate scripts for.")
    parser.add_argument('--sizes', nargs='+', type=int, default=[5,6,7,8,9,10,11,12,13,14,15], help="List of graph sizes.")
    parser.add_argument('--base_path', type=str, default='/home/ataylor2/algorithmic_reasoning/proj_baseline/', help="Base path for the output directories.")
    parser.add_argument('--windows', nargs='+', type=int, default=[1,2,5], help="List of window sizes.")

    # parser.add_argument('--reasoning_strategy', default='Int_Steps_Wndw', choices=["Int_Steps", "IO", "Int_Steps_Wndw"], help="Reasoning strategy to use in the directory structure.")
    # parser.add_argument('--model', default='llama', choices=["llama", "mistral"], help="Model to use.")
    
    args = parser.parse_args()
    
    reasoning_types = ["IO_no_chat"]#["Int_Steps", "IO"]
    chat_types = ["chat", "vanilla", "vanilla_mistral", "chat_mistral"] #"chat", "vanilla", "vanilla_mistral", "chat_mistral"
    algorithms = ["bfs", "dfs", "dijkstra", "floyd_warshall", "mst_prim"]
    
    #5: 3, 5
    #4: 1, 3, 4, 5
    device_mapping = {
        5: 0, #5
        6: 7, #5
        7: 0, #5
        8: 7, #5
        9: 0, #5
        10: 3, #5
        11: 3, #4
        12: 5, #4
        13: 6, #5
        14: 0, #5
        15: 7, #5
        # 20: 0, #4
        # 50: 0, #4s
    }
    
    script_mapping = {
        5: 1, #5
        6: 2, #5
        7: 1, #5
        8: 2, #5
        9: 1, #5
        10: 0, #5
        11: 0, #4
        12: 5, #4
        13: 6, #5
        14: 1, #5
        15: 2, #5
        # 20: 0, #4
        # 50: 0, #4
    }
    
    
    script_paths = {}
    
    for reasoning_type in reasoning_types:
        for chat_type in chat_types:
            for algorithm in algorithms:
                if reasoning_type == "Int_Steps_Wndw":
                    for window in args.windows:
                        script_names = generate_run_scripts(algorithm, args.sizes, args.base_path, reasoning_type, chat_type, device_mapping, window)
                else:
                    script_names = generate_run_scripts(algorithm, args.sizes, args.base_path, reasoning_type, chat_type, device_mapping)
                for graph_size in script_names:
                    
                    if not graph_size in script_paths:
                        script_paths[graph_size] = []
                    script_paths[graph_size].append(script_names[graph_size])
                    
    assigned_scripts = {}
    for graph_size in script_paths:
        if not script_mapping[graph_size] in assigned_scripts:
            assigned_scripts[script_mapping[graph_size]] = []
        assigned_scripts[script_mapping[graph_size]].extend(script_paths[graph_size])
    
    run_script_path = "/home/ataylor2/algorithmic_reasoning/proj_baseline/alg_run_scripts"
    for key_ in assigned_scripts:
        print(f"{key_}: {len(assigned_scripts[key_])}")
        with open(os.path.join(run_script_path, f"run_script_{key_}.sh"), 'w') as f:
            for file_name in assigned_scripts[key_]:
                f.write(f"sh {file_name}\n")