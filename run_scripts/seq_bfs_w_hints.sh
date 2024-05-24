#!/bin/bash

# Set environment variable for CUDA
export CUDA_VISIBLE_DEVICES=4
wandb online

# Loop through graph sizes from 3 to 10 (inclusive)
for graph_size in {6..10}
do
    echo "Starting job for graph size: $graph_size"

    # Execute the Python script with the current graph size
    # Continue on error
    set +e
    python ../train.py ../configs/bfs_w_hints.json $graph_size

    # Optionally, check if the Python script was successful
    # and perform any necessary actions
    if [ $? -eq 0 ]; then
        echo "Job for graph size $graph_size completed successfully."
    else
        echo "Job for graph size $graph_size failed. Moving to the next one."
    fi

    # Reset 'e' option to its default behavior if needed elsewhere in the script
    # set -e
done

echo "All jobs have been processed."
