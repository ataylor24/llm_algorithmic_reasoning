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

def _translate_source_node(source_list):
    return int(np.nonzero(source_list.flatten())[0][0])

def _datapoint_to_dict(dp):
    return {"name":dp.name,
            "location":dp.location,
            "data":dp.data}
def _datapoints_list_to_dict(dp_list):
    dp_dict = {}
    for dp in dp_list:
        dp_dict[dp.name] = _datapoint_to_dict(dp)
    return dp_dict

def _write_data(output_formats, clrs_data_dir, dict_llm_data_dir, clrs_training_data, clrs_validation_data, clrs_testing_data, trans_training_data, trans_validation_data, trans_testing_data):
    #Writing CLRS data
    data_utils.write_clrs_format(os.path.join(clrs_data_dir, "training" + ".pkl"), clrs_training_data)
    data_utils.write_clrs_format(os.path.join(clrs_data_dir, "validation" + ".pkl"), clrs_validation_data)
    data_utils.write_clrs_format(os.path.join(clrs_data_dir, "testing" + ".pkl"), clrs_testing_data)
    
    #Writing LMM data
    for output_format in output_formats:
        llm_data_dir = dict_llm_data_dir[output_format]
        
        if output_format == "llama2":
            data_utils.write_llama_format(llm_data_dir, "training", trans_training_data)
            data_utils.write_llama_format(llm_data_dir, "validation", trans_validation_data)
            data_utils.write_llama_format(llm_data_dir, "testing", trans_testing_data) 
        else:
            raise NotImplementedError(f"Output format {output_format} has not been implemented.")
    
def _bfs_translate_output(list_pred):
    list_out_idxs = [str(node_idx) for node_idx, pred_idx in enumerate(list_pred) if pred_idx != node_idx]
    return f"### Reachable Nodes: {', '.join(list_out_idxs)}" # if len(list_out_idxs) > 0 else "There are no reachable nodes"

def translate_outputs(alg, outputs):
    outputs_dict = _datapoints_list_to_dict(outputs)
    if alg in ["bfs"]:
        # unweighted graph algorithms
        list_out_preds = outputs_dict["pi"]["data"][0]
        list_out = _bfs_translate_output(list_out_preds)
        return list_out
    elif alg in ["dfs"]:
        list_out_preds = outputs_dict["pi"]["data"][0]
        #print (list_out_preds)
        list_out = _bfs_translate_output(list_out_preds)
        return list_out
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
def _translate_inputs(alg, inputs):
    inputs_dict = _datapoints_list_to_dict(inputs)
    # print("------------INPUTS------------")
    # print(inputs_dict)
    if alg in ["bfs"]:
        # unweighted graph algorithms
        algorithm = alg
        list_edge = _translate_unweighted_graph(inputs_dict["adj"]["data"])
        source = _translate_source_node(inputs_dict["s"]["data"])
        return algorithm, list_edge, source
    if alg in ["dfs"]:
        # unweighted graph algorithms
        algorithm = alg
        list_edge = _translate_unweighted_graph(inputs_dict["adj"]["data"])
        #source = _translate_source_node(inputs_dict["A"]["data"])
        return algorithm, list_edge, 1 #Junk source node to fix formatting/parsing issues
    elif alg in ["dka", "bfd"]:
        #potentially weighted graph algorithms
        raise NotImplementedError(f"[WILL BE REPLACED] No input translation functionality has been implemented for {alg}")
    else:
        raise NotImplementedError(f"No input translation functionality has been implemented for {alg}")

def hash_edgelist(edgelist):
    canonicalEdges = sorted([str(sorted(edge)) for edge in edgelist])  # Canonical form and sort
    return hash(",".join(canonicalEdges))  # Convert to unique representation

def _preprocess_hint_matrix(alg, matrix_h):
    ''' For graph-based approaches (ex. BFS), the hint matrices are actually 2D lists.
        The row index position implicitly refers to the node in question, and the
        value at the index depends on the hint type. '''
    if alg in ["bfs"]:
        # unweighted graph algorithms
        list_flat_h = [unflat_h[0] for unflat_h in matrix_h.astype(int).tolist()]
        return list_flat_h
    if alg in ["dfs"]: 
        list_flat_h = [unflat_h[0] for unflat_h in matrix_h.astype(int).tolist()]
        return list_flat_h
    elif alg in ["dka", "bfd"]:
        #potentially weighted graph algorithms
        raise NotImplementedError(f"[WILL BE REPLACED] No hint translation functionality has been implemented for {alg}")
    else:
        raise NotImplementedError(f"No hint translation functionality has been implemented for {alg}")


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

"""
The majority of the work for translation, parsing the hint lists for each attribute into natural language instructions.

--- Desired Structure for DFS Translations: ---
System Prompt:...

Instruction: Please perform the Depth-first Search algorithm for reachability on this graph: Edgelist: [...], Source Node: ... . Perform the algorithm step by step.
Current Stack: [...]
Pop from Stack: ...
Neighborhood of ...: [...]
List all known reachable nodes.

Response: Reachable Nodes: [...]

"""
def get_reachable_nodes (node, edgelist, visited_nodes): 
    reachable_nodes = []
    for e in edgelist: 
        if e[0] == node and e[1] not in visited_nodes: 
            reachable_nodes.append(e[1])
        elif e[1] == node and e[0] not in visited_nodes:
            reachable_nodes.append(e[0])
    return sorted(reachable_nodes)

def _dfs_translate_list_h (neg_edges, edgelist_lookup, list_pred_h, list_color_h, list_source_h):
    reach_stack = []
    reach_stack_prev = [-1]

    hints = []
    
    node_states = [0]*len(list_color_h[0])
    visited = []
    source_state = -1

    for i in range (len(list_color_h)):
        c = list_color_h[i]

        for j in range (len(c)): 
                node_color = c[j]
                if (node_color == [0,1,0] and node_states[j] != 1): #if a node is visited for the first time
                    if (list_source_h[i].index(1.0) != source_state): 
                        hints.append(f"Push Source {j} to stack and consider its connections.")
                        source_state = list_source_h[i].index(1.0)
                    else:
                        hints.append(f"Push Node {j} to stack and consider its connections.")
                    reach_stack.append(j)
                    visited.append(j)
                    node_states[j] = 1

                    hints.append(f"Reachable Nodes from {j}: {get_reachable_nodes(j, edgelist_lookup, visited)}")

                elif (node_color == [0,0,1] and node_states[j] != 2):
                    hints.append(f"Node {j} has no more connections. Pop from Stack.")
                    reach_stack.pop()
                    node_states[j] = 2

        stack_update = f"Current Stack: {reach_stack}"
        hints.append(stack_update)
        if len(hints) >=2 and hints[-1] == hints[-2]: 
            hints.pop()

        # if (list_source_h[0].index(1.0) != source_state): 
        #     source_state = list_source_h[0].index(1.0)
        #     hints.append(f"Push Source {source_state} to stack and consider its connections.")
        #     reach_stack.append(source_state)
        #     #hints.append(f"Current Stack: {list(reach_stack)}")

    # print ("--SOURCES--")
    # for i in range(len(list_source_h)):
    #     
    #     print(source_state)

    return hints

"""
Parses the hint dictionary into relevant attribute lists, and
passes to the translation function for each algorithm. 
"""
def translate_hints(alg, neg_edges, edgelist_lookup, hints):
    hints_dict = _datapoints_list_to_dict(hints)
    #print("------------HINTS------------")
    #print(hints_dict['s'])
    # print(hints_dict)

    if alg in ["bfs"]:
        # unweighted graph algorithms
        list_reach_h = _preprocess_hint_matrix(alg, hints_dict["reach_h"]["data"])
        list_pred_h = _preprocess_hint_matrix(alg, hints_dict["pi_h"]["data"])
        list_h = _bfs_translate_reach_pred_h(neg_edges, edgelist_lookup, list_reach_h, list_pred_h)
        return list_h
    elif alg in ["dfs"]:
        """
        Different information stored in the hint and input matrices for BFS and DFS.
        BFS: {'reach_h', 'pi_h'}
        DFS: {'pi_h', 'color', 'd', 'f', 's_prev', 's', 'u', 'v', 's_last', 'time'}
        
        Alex did a walkthrough on Thursday on each of these attributes.
        pi_h: Immediate parent? Predecessors? 
        color: Indicates status of each node at every iteration! 
        s: Source Node
        """
        #list_reach_h = _preprocess_hint_matrix(alg, hints_dict["reach_h"]["data"])
        list_pred_h = _preprocess_hint_matrix(alg, hints_dict["pi_h"]["data"])
        list_color_h = _preprocess_hint_matrix(alg, hints_dict["color"]["data"])
        list_source_h = _preprocess_hint_matrix(alg, hints_dict["s"]["data"])
        list_h = _dfs_translate_list_h(neg_edges, edgelist_lookup, list_pred_h, list_color_h, list_source_h)
        return list_h
    elif alg in ["dka", "bfd"]:
        #potentially weighted graph algorithms
        raise NotImplementedError(f"[WILL BE REPLACED] No hint translation functionality has been implemented for {alg}")
    else:
        raise NotImplementedError(f"No hint translation functionality has been implemented for {alg}")


"""
Calls all of the other supporting translation functions and writes the data to the output files.
"""
def sample_data(args):
    clrs_training_data = {}
    clrs_validation_data = {}
    clrs_testing_data = {}
    
    trans_training_data = {}
    trans_validation_data = {}
    trans_testing_data = {}
    
    graph_sizes = range(3, args.graph_sizes + 1)
    
    for graph_size in graph_sizes:
        unique_graphs = set()
        clrs_data_dir, dict_llm_data_dir = data_utils.resolve_output_dirs(args.output_dir, args.algorithm, args.output_formats, graph_size)

        training_instances = data_utils.TRAIN_TEST_SPLIT[graph_size][0] if graph_size in data_utils.TRAIN_TEST_SPLIT else args.train_test_split[0]
        evaluation_instances = data_utils.TRAIN_TEST_SPLIT[graph_size][1] if graph_size in data_utils.TRAIN_TEST_SPLIT else args.train_test_split[1]
        
        data_smp, spec = smp.build_sampler(args.algorithm, num_samples=-1, length=graph_size, seed=args.seed)
        # test_smp, spec = smp.build_sampler(args.algorithm, num_samples=evaluation_instances, length=graph_size, seed=args.seed)
    
        data_smp_iter = _iterate_sampler(data_smp, batch_size=1)
        # test_iter = _iterate_sampler(test_smp, batch_size=1)
        
        valid_train_idx = 0
        valid_eval_idx = 0
        
        while valid_train_idx < training_instances:
            train_sample = next(data_smp_iter)

            """
            TRANSLATION ISSUE #1.
            Source node is not in inputs, is in the hints matrix.
            Made this adjustment, and _translate_inputs appears to work as needed. 
            """
            inputs = _translate_inputs(args.algorithm, train_sample.features.inputs)
            # print("-- INPUTS --")
            # print(inputs)
            
            edgelist_hash = hash_edgelist(inputs[1])
            if edgelist_hash in unique_graphs:
                continue
            
            """
            TRANSLATION ISSUE #2. 

            """
            hints = translate_hints(args.algorithm, args.neg_edges, set(inputs[1]), train_sample.features.hints)
            
            """
            TRANSLATION ISSUE #3. 
            """
            outputs = translate_outputs(args.algorithm, train_sample.outputs)

            clrs_training_data[valid_train_idx] = train_sample
            trans_training_data[valid_train_idx] = {
                "inputs": inputs,
                "hints": "\n".join(hints),
                "outputs": outputs
            }
            
            unique_graphs.add(edgelist_hash)
            valid_train_idx += 1
            
        while valid_eval_idx < evaluation_instances:
            test_sample = next(data_smp_iter)
            inputs = _translate_inputs(args.algorithm, test_sample.features.inputs)
            
            edgelist_hash = hash_edgelist(inputs[1])
            if edgelist_hash in unique_graphs:
                continue
            
            hints = translate_hints(args.algorithm, args.neg_edges, set(inputs[1]), test_sample.features.hints)
            outputs = translate_outputs(args.algorithm, test_sample.outputs)

            if valid_eval_idx < evaluation_instances // 2:
                clrs_validation_data[valid_eval_idx] = test_sample
                trans_validation_data[valid_eval_idx] = {
                    "inputs": inputs,
                    "hints": "\n".join(hints),
                    "outputs": outputs
                }
            else:
                test_idx = valid_eval_idx % (evaluation_instances // 2)
                clrs_testing_data[test_idx] = test_sample
                trans_testing_data[test_idx] = {
                    "inputs": inputs,
                    "hints": "\n".join(hints),
                    "outputs": outputs
                }
            
            unique_graphs.add(edgelist_hash)
            valid_eval_idx += 1
        print(f"Sampling complete for graph size: {graph_size}")
        
        _write_data(args.output_formats, clrs_data_dir, dict_llm_data_dir, clrs_training_data, clrs_validation_data, clrs_testing_data, trans_training_data, trans_validation_data, trans_testing_data)
    

"""
Parses arguments for construct_data from the command line
"""
def main():
    args = data_utils.parse_args()
    sample_data(args)
    
if __name__ == "__main__":
    main()
