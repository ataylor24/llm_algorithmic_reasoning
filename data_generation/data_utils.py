import argparse
import os

def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(description="Jointly constructs CLRS data and sequence-to-sequence data.")

    # Add arguments
    parser.add_argument("algorithm", type=str, choices=['bfs', 'dfs', 'dka', 'bfd'], 
                        help="Algorithm must be one of: 'bfs', 'dfs','dka', or 'bfd'.")
    parser.add_argument("-graph_size", "--graph_size", type=int, default=8, help="Number of nodes present in the graphs generated.")
    parser.add_argument("-num_samples", "--num_samples", type=int, default=10, help="Number of data samples to generate.")
    parser.add_argument("-neg_edges", "--neg_edges", type=bool, default=True, help="Include negative edges, ex. '0 is not reachable from 1'.")
    parser.add_argument("-seed", "--seed", type=int, default=100898, help="Random seed used in constructing the CLRS sampler; the default is 10081998.")
    parser.add_argument("-output_dir", "--output_dir", type=str, default="/local2/ataylor2/algorithmic_reasoning", help="Output directory. Will create folders named after the algorithm for which data is generated.")
    parser.add_argument("-train_test_split", "--train_test_split", type=list, default=[3,2], help="Training/Testing split ratios. The Test set will be equally split into Validation and Test.")
    # Parse the arguments
    args = parser.parse_args()

    return args

def resolve_output_dirs(output_dir, algorithm):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    parent_data_dir = os.path.join(output_dir, algorithm)
    if not os.path.exists(parent_data_dir):
        os.mkdir(parent_data_dir)
        
    clrs_data_dir = os.path.join(parent_data_dir, "clrs_data")
    llm_data_dir = os.path.join(parent_data_dir, "llm_data")
    
    if not os.path.exists(clrs_data_dir):
        os.mkdir(clrs_data_dir)
    if not os.path.exists(llm_data_dir):
        os.mkdir(llm_data_dir)
    
    return clrs_data_dir, llm_data_dir
     