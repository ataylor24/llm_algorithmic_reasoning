import os
import samplers as smp
import numpy as np
from collections import deque
import json
import dill
import data_utils
       
def _iterate_sampler(sampler, batch_size):
        while True:
            yield sampler.next(batch_size)
            
def _preprocess_hint_matrix(alg, matrix_h):
    ''' For graph-based approaches (ex. BFS), the hint matrices are actually 2D lists.
        The row index position implicitly refers to the node in question, and the
        value at the index depends on the hint type. '''
    if alg in ["bfs", "dfs"]:
        # unweighted graph algorithms
        list_flat_h = [unflat_h[0] for unflat_h in matrix_h.astype(int).tolist()]
        return list_flat_h
    elif alg in ["dka", "bfd"]:
        #potentially weighted graph algorithms
        raise NotImplementedError(f"[WILL BE REPLACED] No hint translation functionality has been implemented for {alg}")
    else:
        raise NotImplementedError(f"No hint translation functionality has been implemented for {alg}")
        
def _translate_unweighted_graph(adj_matrix):
    adj_matrix = adj_matrix.squeeze()
    rows, cols = adj_matrix.shape

    # Create an empty list to store edges
    edge_list = []

    # Iterate over each cell in the matrix
    for i in range(rows):
        for j in range(i, cols):  # Start from i to avoid duplicate edges
            if i == j:
                continue
            if adj_matrix[i][j] >= 1:  # Check if there's a connection
                edge_list.append((i, j))

    return edge_list

def _translate_source_node(source_list):
    return int(np.nonzero(source_list.flatten())[0][0])

def _bfs_translate_output(list_pred):
    list_out_idxs = [str(node_idx) for node_idx, pred_idx in enumerate(list_pred) if pred_idx != node_idx]
    return f"Reachable Nodes: {', '.join(list_out_idxs)}" if len(list_out_idxs) > 0 else "There are no reachable nodes"

def _bfs_translate_reach_pred_h(neg_edges, edgelist_lookup, list_reach_h, list_pred_h):
    dict_reach_h = {}
    reach_h_queue = []
    visited_ = set()
    for level_h, (reach_h, pred_h) in enumerate(zip(list_reach_h, list_pred_h)):            
        level_h_queue = set()
        # termination condition
        if sum(reach_h) == 0 and sum(pred_h) == 0:
            continue
 
        for node_idx, (reach_f, pred_node_idx) in enumerate(zip(reach_h, pred_h)):
            
            if not pred_node_idx in dict_reach_h:
                dict_reach_h[pred_node_idx] = set()
            
            if reach_f == 1:
                if node_idx != pred_node_idx: 
                    dict_reach_h[pred_node_idx].add((node_idx, pred_node_idx))  
                if not node_idx in visited_:
                    level_h_queue.add(node_idx)
                    visited_.add(node_idx)
        reach_h_queue.append(sorted(list(level_h_queue)))
    
    hints = []
    idx = 0
    bfs_queue = deque(reach_h_queue[0])
    list_node_idxs = [i for i in range(len(list_reach_h[0]))]
    bfs_dequeue = set()
        
    for reach_h_subqueue in reach_h_queue:
        for reach_h in reach_h_subqueue:
            bfs_subqueue = set()
            hints.append(f"Current Queue: {list(bfs_queue)}")
            current_source = bfs_queue.popleft()
            hints.append(f"Pop {current_source} from queue, and consider its connections: {reach_h}")
            
            if neg_edges:
                bfs_dequeue.add(current_source)
            
            if len(dict_reach_h[reach_h]) == 0:
                if idx == 0:
                    hints.append(f"Source {reach_h} has no connections, therefore we terminate.")
                else:
                    suffix = ", so we move the the next queue element." if len(bfs_queue) > 0 else ", and since the queue is empty, we terminate."
                    hints.append(f"{reach_h} has no additional connections{suffix}")
                continue
            
            #order the hints by placing the lowest node idx first
            dict_reach_h[reach_h] = sorted(list(dict_reach_h[reach_h]))
            
            for node_idx, pred_node_idx in dict_reach_h[reach_h]:
                bfs_subqueue.add(node_idx)
                hints.append(f"{node_idx} is reachable from {pred_node_idx}.")
            if neg_edges:
                for node_idx in list_node_idxs:
                    if node_idx == pred_node_idx or (node_idx, pred_node_idx) in bfs_subqueue: 
                        continue
                    if node_idx not in bfs_subqueue:
                        if ((node_idx, pred_node_idx) in edgelist_lookup or
                            (pred_node_idx, node_idx) in edgelist_lookup) and node_idx in bfs_dequeue:
                            # Node is reachable but has already been reached by a prior node
                            hints.append(f"{node_idx} is reachable from {pred_node_idx}, but has been reached already.")
                        else:
                            hints.append(f"{node_idx} is not reachable from {pred_node_idx}.")
                    # No action required if node_idx is in bfs_subqueue

            bfs_queue.extend(sorted(list(bfs_subqueue)))
            idx += 1
            
    return hints

def _datapoint_to_dict(dp):
    return {"name":dp.name,
            "location":dp.location,
            "data":dp.data}

def _datapoints_list_to_dict(dp_list):
    dp_dict = {}
    for dp in dp_list:
        dp_dict[dp.name] = _datapoint_to_dict(dp)
    return dp_dict

def _write_data(clrs_data_dir, llm_data_dir, clrs_training_data, clrs_validation_data, clrs_testing_data, trans_training_data, trans_validation_data, trans_testing_data):
    def _json_write(data, data_dir, file_prefix):
        json.dump(data, open(os.path.join(data_dir, file_prefix + ".json"), "w"))
    
    def _pickle_compress(data, data_dir, file_prefix):
        dill.dump(data, open(os.path.join(data_dir, file_prefix + ".pkl"), 'wb'))
    
    #Writing CLRS data
    _pickle_compress(clrs_training_data, clrs_data_dir, "training")
    _pickle_compress(clrs_validation_data, clrs_data_dir, "validation")
    _pickle_compress(clrs_testing_data, clrs_data_dir, "testing")
    
    #Writing LMM data
    _json_write(trans_training_data, llm_data_dir, "training")
    _json_write(trans_validation_data, llm_data_dir, "validation")
    _json_write(trans_testing_data, llm_data_dir, "testing")

def translate_outputs(alg, outputs):
    outputs_dict = _datapoints_list_to_dict(outputs)

    if alg in ["bfs", "dfs"]:
        # unweighted graph algorithms
        list_out_preds = outputs_dict["pi"]["data"][0]
        list_out = _bfs_translate_output(list_out_preds)
        return list_out
    elif alg in ["dka", "bfd"]:
        #potentially weighted graph algorithms
        raise NotImplementedError(f"[WILL BE REPLACED] No hint translation functionality has been implemented for {alg}")
    else:
        raise NotImplementedError(f"No hint translation functionality has been implemented for {alg}")


def translate_hints(alg, neg_edges, edgelist_lookup, hints):
    hints_dict = _datapoints_list_to_dict(hints)

    if alg in ["bfs", "dfs"]:
        # unweighted graph algorithms
        list_reach_h = _preprocess_hint_matrix(alg, hints_dict["reach_h"]["data"])
        list_pred_h = _preprocess_hint_matrix(alg, hints_dict["pi_h"]["data"])
        list_h = _bfs_translate_reach_pred_h(neg_edges, edgelist_lookup, list_reach_h, list_pred_h)
        return list_h
    elif alg in ["dka", "bfd"]:
        #potentially weighted graph algorithms
        raise NotImplementedError(f"[WILL BE REPLACED] No hint translation functionality has been implemented for {alg}")
    else:
        raise NotImplementedError(f"No hint translation functionality has been implemented for {alg}")


def _translate_inputs(alg, inputs):
    inputs_dict = _datapoints_list_to_dict(inputs)

    if alg in ["bfs", "dfs"]:
        # unweighted graph algorithms
        list_edge = _translate_unweighted_graph(inputs_dict["adj"]["data"])
        source = _translate_source_node(inputs_dict["s"]["data"])
        return list_edge, source
    elif alg in ["dka", "bfd"]:
        #potentially weighted graph algorithms
        raise NotImplementedError(f"[WILL BE REPLACED] No input translation functionality has been implemented for {alg}")
    else:
        raise NotImplementedError(f"No input translation functionality has been implemented for {alg}")

def sample_data(args):
    clrs_training_data = {}
    clrs_validation_data = {}
    clrs_testing_data = {}
    
    trans_training_data = {}
    trans_validation_data = {}
    trans_testing_data = {}
    
    clrs_data_dir, llm_data_dir = data_utils.resolve_output_dirs(args.output_dir, args.algorithm)
    
    train_smp, spec = smp.build_sampler(args.algorithm, num_samples=args.num_samples, length=args.graph_size, seed=args.seed)
    test_smp, spec = smp.build_sampler(args.algorithm, num_samples=args.num_samples, length=args.graph_size, seed=args.seed)

    train_iter = _iterate_sampler(train_smp, batch_size=1)
    test_iter = _iterate_sampler(test_smp, batch_size=1)
    
    for i in range(args.train_test_split[0]):
        train_sample = next(train_iter)
        inputs = _translate_inputs(args.algorithm, train_sample.features.inputs)
        hints = translate_hints(args.algorithm, args.neg_edges, set(inputs[0]), train_sample.features.hints)
        outputs = translate_outputs(args.algorithm, train_sample.outputs)

        clrs_training_data[i] = train_sample
        trans_training_data[i] = {
            "inputs": inputs,
            "hints": "\n".join(hints),
            "outputs": outputs
        }
    
    for i in range(args.train_test_split[1]):
        test_sample = next(test_iter)
        inputs = _translate_inputs(args.algorithm, test_sample.features.inputs)
        hints = translate_hints(args.algorithm, args.neg_edges, set(inputs[0]), test_sample.features.hints)
        outputs = translate_outputs(args.algorithm, test_sample.outputs)
        if i < args.train_test_split[1] // 2:
            clrs_validation_data[i] = test_sample
            trans_validation_data[i] = {
                "inputs": inputs,
                "hints": "\n".join(hints),
                "outputs": outputs
            }
        else:
            clrs_testing_data[i] = test_sample
            trans_testing_data[i] = {
                "inputs": inputs,
                "hints": "\n".join(hints),
                "outputs": outputs
            }
    
    _write_data(clrs_data_dir, llm_data_dir, clrs_training_data, clrs_validation_data, clrs_testing_data, trans_training_data, trans_validation_data, trans_testing_data)
    
def main():
    args = data_utils.parse_args()
    sample_data(args)
    
if __name__ == "__main__":
    main()
