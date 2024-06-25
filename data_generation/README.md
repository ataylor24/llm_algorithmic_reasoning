# Construct Data Module

## Overview
The `construct_data.py` script is part of a larger project focused on generating and preprocessing datasets for various graph algorithms. It supports algorithms like Breadth-First Search (BFS), Depth-First Search (DFS), Dijkstra's algorithm, Floyd-Warshall algorithm, and Prim's Minimum Spanning Tree (MST). The script converts graph data into formats suitable for training machine learning models and saves the datasets in multiple formats, such as CLRS and LLM-compatible formats.

## Adding New Algorithms
To add support for a new graph algorithm, follow these steps:

1. **Define Preprocessing Functions**: Implement preprocessing functions for the new algorithm. These functions should handle the translation of input adjacency matrices and hint matrices specific to the algorithm.

2. **Translate Inputs**: Update the `_translate_inputs` function to handle the new algorithm by converting its specific inputs into a standardized format.

3. **Translate Hints**: Implement hint translation functions for the new algorithm. Add these functions to the `translate_hints` function to process hints specific to the algorithm.

4. **Translate Outputs**: Create output translation functions for the new algorithm and update the `translate_outputs` function to format the outputs accordingly.

5. **Sampling Data**: Ensure the `sample_data` function includes the new algorithm in its sampling process, generating the required data samples and hints.

6. **Modify Argument Parsing**: Update the `parse_args` function to include the new algorithm in the list of supported algorithms.

### Example of Adding a New Algorithm
Assuming the new algorithm is "A-Star":

1. **Preprocessing Functions**:
   ```python
   def _translate_a_star_inputs(adj_matrix):
       # Implement the function to translate A-Star inputs
       pass

   def _translate_a_star_hints(hints_dict, source):
       # Implement the function to translate A-Star hints
       pass

   def _translate_a_star_outputs(outputs):
       # Implement the function to translate A-Star outputs
       pass
   ```

2. **Update `_translate_inputs`**:
   ```python
   def _translate_inputs(alg, inputs):
       # Existing code ...
       elif alg == "a_star":
           algorithm = alg
           list_edge_with_weights = _translate_a_star_inputs(inputs_dict["adj"]["data"])
           source = _translate_source_node(inputs_dict["s"]["data"])
           return algorithm, list_edge_with_weights, source
       else:
           raise NotImplementedError(f"No input translation functionality has been implemented for {alg}")
   ```

3. **Update `translate_hints`**:
   ```python
   def translate_hints(alg, neg_edges, edgelist_lookup, hints, source=None):
       # Existing code ...
       elif alg == "a_star":
           return _translate_a_star_hints(hints_dict, source)
       else:
           raise NotImplementedError(f"No hint translation functionality has been implemented for {alg}")
   ```

4. **Update `translate_outputs`**:
   ```python
   def translate_outputs(alg, outputs, final_d=None):
       # Existing code ...
       elif alg == "a_star":
           return _translate_a_star_outputs(outputs)
       else:
           raise NotImplementedError(f"No output translation functionality has been implemented for {alg}")
   ```

5. **Update `parse_args`**:
   ```python
   def parse_args():
       parser = argparse.ArgumentParser(description="Jointly constructs CLRS data and sequence-to-sequence data.")
       parser.add_argument("algorithm", type=str, choices=['bfs', 'dfs', 'dijkstra', 'floyd_warshall', 'mst_prim', 'a_star'], 
                           help="Algorithm must be one of: 'bfs', 'dfs', 'dijkstra', 'floyd_warshall', 'mst_prim', 'a_star'.")
       # Existing code ...
       return args
   ```

## Usage
To use the `construct_data.py` script, run it from the command line with appropriate arguments. The script supports various command-line options to customize the data generation process, including the algorithm type, graph sizes, number of samples, and output formats.

Example:
```sh
python construct_data.py bfs --graph_sizes [5,6,7] --num_samples 100 --output_dir /path/to/output
```

Refer to the main README of the project for installation instructions and a summary of the overall codebase.
