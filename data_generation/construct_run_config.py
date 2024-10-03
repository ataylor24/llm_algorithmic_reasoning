import yaml
import os
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Update YAML files with dataset mixer and output directory.")
# parser.add_argument('--algorithm', type=str, default="dfs", help="Algorithm to use (default: bfs)")
# parser.add_argument('--reasoning_type', type=str, default="Int_Steps_Wndw", help="Reasoning type to use (default: Int_Steps)")
parser.add_argument('--base_path', type=str, default="/home/ataylor2/algorithmic_reasoning/", help="Base path for the project (default: /home/ataylor2/algorithmic_reasoning/)")
parser.add_argument('--train_batch_size', type=int, default=1, help="Batch size for training")
parser.add_argument('--graph_sizes', type=int, nargs='+', default=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 50], help="List of graph sizes (default: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 50])")
# parser.add_argument('--chat_type', type=str, default="chat", help="chat (llama) or chat_mistral (mistral)")
parser.add_argument('--r_alpha', type=int, default=16, help="r and alpha hyperparameters")

args = parser.parse_args()
reasoning_types = ["IO_no_chat"]#["Int_Steps", "IO", "IO_no_chat"]
chat_types = ["chat", "vanilla", "vanilla_mistral", "chat_mistral"] #"chat", "vanilla", "vanilla_mistral", "chat_mistral"
algorithms = ["bfs", "dfs", "dijkstra", "floyd_warshall", "mst_prim"]
# algorithm = args.algorithm
# chat_type = args.chat_type
# reasoning_type = args.reasoning_type if not args.reasoning_type != "Int_Steps_Wndw" and args.window == -1 else args.reasoning_type + f"_{args.window}"
base_path = args.base_path
graph_sizes = args.graph_sizes
train_batch_size = args.train_batch_size

models = {
    "chat": ("meta-llama/Meta-Llama-3-8B-Instruct", "Llama3-8b-instruct-sft-qlora"),
    "chat_mistral": ("mistralai/Mistral-7B-Instruct-v0.3", "Mistral-7b-instruct-sft-qlora"),
    "vanilla": ("meta-llama/Meta-Llama-3-8B", "Llama3-8b-sft-qlora"),
    "vanilla_mistral": ("mistralai/Mistral-7B-v0.3", "Mistral-7b-sft-qlora"),
}


r_alpha = args.r_alpha

base_yaml = os.path.join(base_path, "proj_baseline/recipes/algorithmic_reasoning/sft/config_qlora.yaml")

# Load the existing YAML file
with open(base_yaml, 'r') as file:
    config = yaml.safe_load(file)
for algorithm in algorithms:
    for reasoning_type in reasoning_types:
        for chat_type in chat_types:
            # Update the dataset_mixer and output_dir fields for each graph size
            for graph_size in graph_sizes:
                dataset_path = f"/local/ataylor2/algorithmic_reasoning/{algorithm}/graph_size_{graph_size}/llm_data/{chat_type}/{reasoning_type}"
                config['dataset_mixer'] = {
                    dataset_path: 1.0
                }
                config['output_dir'] = dataset_path
                config['per_device_train_batch_size'] = train_batch_size
                config['lora_alpha'] = r_alpha if algorithm != "bfs" else 8
                config['lora_r'] = r_alpha if algorithm != "bfs" else 8
                config['model_name_or_path'] = models[chat_type][0]
                config['hub_model_id'] = models[chat_type][1]

                # Create the directory if it does not exist
                os.makedirs(dataset_path, exist_ok=True)
                print(dataset_path)
                # Save the updated YAML file
                with open(os.path.join(dataset_path, "config_qlora.yaml"), 'w') as file:
                    yaml.dump(config, file, default_flow_style=False)

print("YAML files updated successfully.")
