# LLM-CLRS Graph Reasoning Benchmark 
Data generation is an adaption of the CLRS benchmark, which can be found here: https://github.com/google-deepmind/clrs

Seed used = 100898

## Introduction

The LLM-CLRS Graph Reasoning Benchmark is designed to evaluate the performance of large language models (LLMs) on classical graph algorithms using intermediate steps. Despite advances in LLMs, they exhibit significant limitations in structured, multistep reasoning tasks, particularly those involving explicit graph structures. Our benchmark addresses this gap by evaluating state-of-the-art LLMs on five fundamental algorithms: BFS, DFS, Dijkstra's, Floyd-Warshall, and Prim's MST.

## Features

- **Comprehensive Benchmark:** Evaluates LLM performance on classical graph algorithms.
- **Intermediate Steps Evaluation:** Focuses on the accuracy of intermediate reasoning steps.
- **Multiple Algorithms:** Includes BFS, DFS, Dijkstra's, Floyd-Warshall, and Prim's MST.
- **Advanced Prompting Techniques:** Explores advanced prompting techniques and algorithmic instructions.

## Installation

### Prerequisites

- Python 3.10 or higher

### Clone the Repository

```bash
git clone https://github.com/yourusername/LLM-CLRS-Graph-Reasoning-Benchmark.git
cd LLM-CLRS-Graph-Reasoning-Benchmark
```

### Create a Conda Environment

To create a Conda environment with the required dependencies, run the following command:

```bash
conda env create --file environment.yml
```

This will create a new Conda environment with all the dependencies specified in the `environment.yml` file.

### Activate the Conda Environment

Activate the newly created environment using:

```bash
conda activate <your_environment_name>
```

Replace `<your_environment_name>` with the name specified in the `environment.yml` file.


## Usage

### Running the Benchmark

To run the benchmark on the included algorithms:

```bash
python run_benchmark.py --algorithm bfs
python run_benchmark.py --algorithm dfs
python run_benchmark.py --algorithm dijkstra
python run_benchmark.py --algorithm floyd-warshall
python run_benchmark.py --algorithm mst-prim
```

### Input and Output

Each algorithm requires specific input formats:

- **BFS and DFS:** Requires an adjacency list or matrix.
- **Dijkstra's and Floyd-Warshall:** Requires a weight matrix.
- **Prim's MST:** Requires an adjacency matrix with weights.

The output will include the results of the algorithmic execution, intermediate steps, and performance metrics.

### Configuration

You can customize the benchmark settings using the configuration file `config.yaml`.

## Performance Metrics

The benchmark evaluates the following metrics:

- **Exact Match Accuracy:** Measures the correctness of the final output.
- **Intermediate Steps Accuracy:** Evaluates the accuracy of intermediate steps.

## Results Interpretation

- **Exact Match Accuracy:** Indicates the percentage of correct final outputs.
- **Intermediate Steps Accuracy:** Provides insight into the reasoning process and where the model might be making errors.

## Contributing

We welcome contributions to improve this benchmark. Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact Information

For questions or feedback, please open an issue or contact us at [your-email@example.com].

---

Thank you for using the LLM-CLRS Graph Reasoning Benchmark! We hope this benchmark helps advance the understanding and capabilities of large language models in structured reasoning tasks.

