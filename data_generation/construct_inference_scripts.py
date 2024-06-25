import os
import argparse

def generate_run_scripts(target_task, reasoning_strategy, graph_size, split_to_use, mode, inference_engine, llm_name, device_id, chat_suffix):
    outer_directory = os.path.join("/home/ataylor2/algorithmic_reasoning/proj_baseline/", 'alg_inference_scripts')
    os.makedirs(outer_directory, exist_ok=True)
    model_tag = f'_{llm_name}' if llm_name != 'meta-llama/Meta-Llama-3-8B' else ''
    directory = os.path.join(outer_directory, f"{target_task}_{reasoning_strategy}{model_tag}_scripts") if chat_suffix == 'None' else os.path.join(outer_directory, f"{target_task}_{reasoning_strategy}{model_tag}{chat_suffix}_scripts")
    os.makedirs(directory, exist_ok=True)

    script_template = """export CUDA_VISIBLE_DEVICES={device_id}
export HF_HOME=/local2/ataylor2/algorithmic_reasoning/cache
export HF_TOKEN={huggingface_key}
{open_ai_key}
python ../inference_bench.py --target_task {target_task} --reasoning_strategy {reasoning_strategy} --graph_size {graph_size} --split_to_use {split_to_use} --mode {mode} --inference_engine {inference_engine} --llm_name {llm_name} --chat_suffix {chat_suffix}

"""

    device_id = device_id

        
    
    script_content = script_template.format(
        open_ai_key=open_ai_key,
        device_id=device_id,
        target_task=target_task, 
        reasoning_strategy=reasoning_strategy, 
        graph_size=graph_size, 
        split_to_use=split_to_use, 
        mode=mode, 
        inference_engine=inference_engine, 
        llm_name=llm_name,
        chat_suffix=chat_suffix
    )
    script_filename = os.path.join(directory, f"run_{target_task}_size_{graph_size}.sh")
    
    with open(script_filename, 'w') as script_file:
        script_file.write(script_content)
    print(f"Generated script: {script_filename}")

    # Increment the device_id, wrapping around to 0 after 6
    # device_id = 1#(device_id + 1) % 7
    # Increment the port
  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate inference scripts for different algorithms and graph sizes.")
    parser.add_argument('--target_task', type=str, default='bfs', choices=['bfs', 'dfs', 'dijkstra', 'floyd_warshall', 'mst_prim'], help='Target task to perform')
    parser.add_argument('--reasoning_strategy', type=str, default='Int_Steps', help='Reasoning strategy to use')
    parser.add_argument('--graph_size', type=str, default='5', choices=['5','6','7','8','9','10','11','12','13','14','15','20','50'], help='Size of the graph')
    parser.add_argument('--split_to_use', type=str, default='evaluate', help='Data split to use (train/test)')
    parser.add_argument('--mode', type=str, default='inference', help='Mode of operation (seq_gen/inference)')
    parser.add_argument('--inference_engine', type=str, default='hf', choices=['hf', 'openai', 'vllm'], help='Inference engine to use (hf/openai/vllm)')
    parser.add_argument('--llm_name', type=str, default='meta-llama/Meta-Llama-3-8B', choices=['meta-llama/Meta-Llama-3-8B', 'gpt-4o'], help='Name of the language model to use')
    parser.add_argument('--from_saved', action='store_true', help='Load from saved model or not')
    parser.add_argument('--device_id', type=int, default=0, help='Load from saved model or not')
    parser.add_argument('--chat_suffix', type=str, default='None', help='helps to rerun models in non-default directories')
    
    args = parser.parse_args()
    
    generate_run_scripts(args.target_task, args.reasoning_strategy, args.graph_size, args.split_to_use, args.mode, args.inference_engine, args.llm_name, args.device_id, args.chat_suffix)
