import os
import argparse

def generate_run_scripts(alg, sizes, base_path, reasoning_strategy, model):
    outer_directory = os.path.join(base_path, 'alg_run_scripts')
    os.makedirs(outer_directory, exist_ok=True)
    model_tag = f'_rerun_ralpha_32' if model == 'mistral' else ''
    directory = os.path.join(outer_directory, f"{alg}_{reasoning_strategy}{model_tag}_scripts")
    os.makedirs(directory, exist_ok=True)
    print(model_tag)
    script_template = """export CUDA_VISIBLE_DEVICES={device_id}
export HF_HOME=/local2/ataylor2/algorithmic_reasoning/cache
export HF_TOKEN=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
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file {base_path}recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 --main_process_port={port} {base_path}scripts/run_sft.py /local/ataylor2/algorithmic_reasoning/{alg}/graph_size_{size}/llm_data/chat{model_tag}/{reasoning_strategy}/config_qlora.yaml --load_in_4bit=true
"""

    device_id = 0
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
        
    for size in sizes:
        script_content = script_template.format(
            alg=alg, 
            size=size, 
            device_id=device_id, 
            port=port, 
            reasoning_strategy=reasoning_strategy,
            base_path=base_path,
            model_tag=model_tag
        )
        script_filename = os.path.join(directory, f"run_{alg}_size_{size}.sh")
        
        with open(script_filename, 'w') as script_file:
            script_file.write(script_content)
        print(f"Generated script: {script_filename}")

        # Increment the device_id, wrapping around to 0 after 6
        device_id = 1#(device_id + 1) % 7
        # Increment the port
        port += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate run scripts for different algorithms and graph sizes.")
    parser.add_argument('--alg', default='bfs', choices=['bfs', 'dfs', 'dijkstra', 'floyd_warshall', 'mst_prim'], help="Algorithm to generate scripts for.")
    parser.add_argument('--sizes', nargs='+', type=int, default=[5,6,7,8,9,10,11,12,13,14,15,20,50], help="List of graph sizes.")
    parser.add_argument('--base_path', type=str, default='/home/ataylor2/algorithmic_reasoning/proj_baseline/', help="Base path for the output directories.")
    parser.add_argument('--reasoning_strategy', default='Int_Steps', choices=["Int_Steps", "IO"], help="Reasoning strategy to use in the directory structure.")
    parser.add_argument('--model', default='mistral', choices=["llama", "mistra"], help="Model to use.")
    
    args = parser.parse_args()
    
    generate_run_scripts(args.alg, args.sizes, args.base_path, args.reasoning_strategy, args.model)
