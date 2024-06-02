import yaml
import os
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Update YAML files with dataset mixer and output directory.")
parser.add_argument('--algorithm', type=str, default="bfs", help="Algorithm to use (default: bfs)")
parser.add_argument('--reasoning_type', type=str, default="Int_Steps", help="Reasoning type to use (default: Int_Steps)")
parser.add_argument('--base_path', type=str, default="/home/ataylor2/algorithmic_reasoning/", help="Base path for the project (default: /home/ataylor2/algorithmic_reasoning/)")
parser.add_argument('--train_batch_size', type=int, default=4, help="Batch size for training")
parser.add_argument('--graph_sizes', type=int, nargs='+', default=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 50], help="List of graph sizes (default: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 50])")

args = parser.parse_args()

algorithm = args.algorithm
reasoning_type = args.reasoning_type
base_path = args.base_path
graph_sizes = args.graph_sizes
train_batch_size = args.train_batch_size

base_yaml = os.path.join(base_path, "proj_baseline/recipes/algorithmic_reasoning/sft/config_qlora.yaml")

# Load the existing YAML file
with open(base_yaml, 'r') as file:
    config = yaml.safe_load(file)

# Update the dataset_mixer and output_dir fields for each graph size
for graph_size in graph_sizes:
    dataset_path = f"/local2/ataylor2/algorithmic_reasoning/{algorithm}/graph_size_{graph_size}/llm_data/chat/{reasoning_type}/run1"
    config['dataset_mixer'] = {
        dataset_path: 1.0
    }
    config['output_dir'] = dataset_path
    config['per_device_train_batch_size'] = train_batch_size

    # Create the directory if it does not exist
    os.makedirs(dataset_path, exist_ok=True)

    # Save the updated YAML file
    with open(os.path.join(dataset_path, "config_qlora.yaml"), 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

print("YAML files updated successfully.")
