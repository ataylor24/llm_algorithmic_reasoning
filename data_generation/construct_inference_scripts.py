import os
import argparse

def generate_inf_scripts(open_ai_key, huggingface_key, target_task, reasoning_strategy, graph_size, split_to_use, mode, inference_engine, llm_name, device_id, chat_type):
    outer_directory = os.path.join("/home/ataylor2/algorithmic_reasoning/proj_baseline/", 'alg_inference_scripts')
    os.makedirs(outer_directory, exist_ok=True)
    model_tag = f'_{llm_name.split("/")[-1]}' if llm_name != 'meta-llama/Meta-Llama-3-8B-Instruct' else ''
    directory = os.path.join(outer_directory, f"{target_task}_{reasoning_strategy}{model_tag}_scripts") #if chat_suffix == 'None' else os.path.join(outer_directory, f"{target_task}_{reasoning_strategy}{model_tag}{chat_suffix}_scripts")
    os.makedirs(directory, exist_ok=True)
    open_ai_secret_key = f"export OPENAI_API_KEY={open_ai_key}" if llm_name == "gpt-4o" else ""
    script_template = """export CUDA_VISIBLE_DEVICES={device_id}
export HF_HOME=/local2/ataylor2/algorithmic_reasoning/cache
export HF_TOKEN={huggingface_key}
{open_ai_secret_key}
python ../inference_bench.py --target_task {target_task} --reasoning_strategy {reasoning_strategy} --graph_size {graph_size} --split_to_use {split_to_use} --mode {mode} --inference_engine {inference_engine} --llm_name {llm_name} --chat_type {chat_type}

"""

    device_id = device_id

        
    
    script_content = script_template.format(
        open_ai_secret_key=open_ai_secret_key,
        huggingface_key=huggingface_key,
        device_id=device_id,
        target_task=target_task, 
        reasoning_strategy=reasoning_strategy, 
        graph_size=graph_size, 
        split_to_use=split_to_use, 
        mode=mode, 
        inference_engine=inference_engine, 
        llm_name=llm_name,
        chat_type=chat_type
    )
    script_filename = os.path.join(directory, f"run_{target_task}_size_{graph_size}.sh")
    
    with open(script_filename, 'w') as script_file:
        script_file.write(script_content)
    # print(f"Generated script: {script_filename}")
    
    return script_filename

    # Increment the device_id, wrapping around to 0 after 6
    # device_id = 1#(device_id + 1) % 7
    # Increment the port
  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate inference scripts for different algorithms and graph sizes.")
    # parser.add_argument('--target_task', type=str, default='bfs', choices=['bfs', 'dfs', 'dijkstra', 'floyd_warshall', 'mst_prim'], help='Target task to perform')
    # parser.add_argument('--reasoning_strategy', type=str, default='Int_Steps', help='Reasoning strategy to use')
    parser.add_argument('--targeted_job', type=bool, default=False, help='Job targeted at specific algorithm/graph size combinations')
    parser.add_argument('--sizes', nargs='+', type=int, default=[5,6,7,8,9,10,11,12,13,14,15], help="List of graph sizes.")
    parser.add_argument('--split_to_use', type=str, default='evaluate', help='Data split to use (train/test)')
    parser.add_argument('--mode', type=str, default='inference', help='Mode of operation (seq_gen/inference)')
    parser.add_argument('--inference_engine', type=str, default='hf', choices=['hf', 'openai', 'vllm'], help='Inference engine to use (hf/openai/vllm)')
    # parser.add_argument('--llm_name', type=str, default='meta-llama/Meta-Llama-3-8B', choices=['meta-llama/Meta-Llama-3-8B', 'gpt-4o'], help='Name of the language model to use')
    parser.add_argument('--from_saved', action='store_true', help='Load from saved model or not')
    parser.add_argument('--device_id', type=int, default=0, help='Device to assign inference to')
    parser.add_argument('--chat_suffix', type=str, default='None', help='helps to rerun models in non-default directories')
    parser.add_argument('--hf_token', type=str, default='[Insert Here]', help='Huggingface Token')
    parser.add_argument('--openai_key', type=str, default='[Insert Here]', help='OpenAI Secret Key')
    
    
    args = parser.parse_args()
    
    targeted_job = args.targeted_job
    
    reasoning_types = ["IO_nc_w_IS", "IO_w_IS_no_tf", "IO_no_tf"] #["IO_w_IS", "IO_no_chat", "Int_Steps", "IO"]
    llm_names = [("chat", "meta-llama/Meta-Llama-3-8B-Instruct")]#[("vanilla", "meta-llama/Meta-Llama-3-8B"), ("chat", "meta-llama/Meta-Llama-3-8B-Instruct"), ("chat_mistral", "mistralai/Mistral-7B-Instruct-v0.3"), ("vanilla_mistral", "mistralai/Mistral-7B-v0.3")]#[('chat_gpt', 'gpt-4o')]
    # chat_types = ["vanilla", "vanilla_mistral", "chat_mistral"] #"chat", "vanilla", "vanilla_mistral", "chat_mistral"
    algorithms = ["bfs", "dfs", "dijkstra", "mst_prim", "floyd_warshall"] 
    
    #5: 3, 5
    #4: 1, 3, 4, 5
    device_mapping = {
        5: 0, #5c
        6: 1, #5
        7: 0, #5
        8: 1, #5
        9: 1, #5
        10: 0, #5
        11: 2, #5
        12: 3, #5
        13: 4, #5
        14: 5, #5
        15: 6, #5
        # 20: 0, #4
        # 50: 0, #4
    }
    
    script_mapping = {
        5: 0, #5c
        6: 1, #5
        7: 0, #5
        8: 1, #5
        9: 1, #5
        10: 0, #5
        11: 2, #5
        12: 3, #5
        13: 4, #5
        14: 5, #5
        15: 6, #5
        # 20: 0, #4
        # 50: 0, #4
    }
    
    script_paths = {}
    
    for reasoning_type in reasoning_types:
        for chat_type, llm_name in llm_names:
            for algorithm in algorithms:
                for graph_size in args.sizes:
                    if targeted_job:
                        if algorithm == "bfs" and not graph_size in [9, 13, 14]:
                            continue
                        if algorithm == "dfs" and not graph_size in [7, 9, 13, 14]:
                            continue
                        if algorithm == "dijkstra" and not graph_size in [7, 8, 9, 10, 13, 14, 15]:
                            continue
                        if algorithm == "floyd_warshall" and not graph_size in [7, 8, 9, 10, 12, 13, 14, 15]:
                            continue
                        if algorithm == "mst_prim" and not graph_size in [7, 8, 9, 10, 12, 13, 14, 15]:
                            continue
                    
                    script_name = generate_inf_scripts(args.openai_key, args.hf_token, algorithm, reasoning_type, graph_size, args.split_to_use, args.mode, args.inference_engine, llm_name, device_mapping[graph_size], chat_type)
                
                    if not graph_size in script_paths:
                        script_paths[graph_size] = []
                    script_paths[graph_size].append(script_name)
                    
    assigned_scripts = {}
    for graph_size in script_paths:
        if not script_mapping[graph_size] in assigned_scripts:
            assigned_scripts[script_mapping[graph_size]] = []
        assigned_scripts[script_mapping[graph_size]].extend(script_paths[graph_size])
    
    run_script_path = "/home/ataylor2/algorithmic_reasoning/proj_baseline/alg_inference_scripts"
    for key_ in assigned_scripts:
        print(f"{key_}: {len(assigned_scripts[key_])}")
        with open(os.path.join(run_script_path, f"inf_script_{key_}.sh"), 'w') as f:
            for file_name in assigned_scripts[key_]:
                f.write(f"sh {file_name}\n")
                
