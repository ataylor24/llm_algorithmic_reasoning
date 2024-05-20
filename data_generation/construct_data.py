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
    # print(f"Preprocessing hint matrix for algorithm: {alg}")
    # print(f"Type of matrix_h: {type(matrix_h)}")
    # print(f"Shape of matrix_h: {matrix_h.shape if isinstance(matrix_h, np.ndarray) else 'N/A'}")
    
    # if not isinstance(matrix_h, np.ndarray):
    #     raise TypeError(f"Expected matrix_h to be a numpy array, but got {type(matrix_h)}")
    
    # if matrix_h.ndim != 2:
    #     raise ValueError(f"Expected matrix_h to be a 2D matrix, but got an array with {matrix_h.ndim} dimensions")
    
    if alg in ["bfs"]:
        # unweighted graph algorithms
        list_flat_h = [unflat_h[0] for unflat_h in matrix_h.astype(int).tolist()]
        print(f"Processed list_flat_h: {list_flat_h}")
        return list_flat_h
    elif alg in ["dfs"]:
        # unweighted graph algorithms
        list_flat_h = [unflat_h[0] for unflat_h in matrix_h.astype(int).tolist()]
        # print(f"Processed list_flat_h: {list_flat_h}")
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
    return f"### Reachable Nodes: {', '.join(list_out_idxs)}"# if len(list_out_idxs) > 0 else "There are no reachable nodes"

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


#most recent function i worked on with alex
# def _dfs_translate_reach_pred_h(neg_edges, edgelist_lookup, list_pred_h, list_color_h, list_discovery_h, 
#                                 list_final_h, list_s_prev_h, list_s_h, 
#                                 list_source_h, list_target_h, list_s_last_h, list_time):
#     dict_reach_h = {}
#     visited_ = set()
#     reachable_from_node = {}  # Dictionary to keep track of reachable nodes from each node

#     # Initialize the reachability dictionary based on predecessor history
#     for level_h, pred_h in enumerate(list_pred_h):
#         for node_idx, pred_node_idx in enumerate(pred_h):
#             if pred_node_idx != node_idx:  # Ensure we don't include self-loops unless explicitly defined
#                 if pred_node_idx not in dict_reach_h:
#                     dict_reach_h[pred_node_idx] = set()
#                 dict_reach_h[pred_node_idx].add(node_idx)

#     # Initialize the stack with the first set of reachable nodes from list_s_h (start nodes)
#     reach_h_stack = set()
#     for source_nodes in list_s_h:
#         if isinstance(source_nodes, np.ndarray) or isinstance(source_nodes, list):
#             for node_idx, source in enumerate(source_nodes):
#                 if source == 1:
#                     reach_h_stack.add(node_idx)
#                     visited_.add(node_idx)
#         else:
#             raise ValueError(f"Unexpected structure in list_s_h: {source_nodes}")

#     hints = []
#     dfs_stack = list(sorted(reach_h_stack))  # Start with lexicographically smallest
#     list_node_idxs = [i for i in range(len(list_pred_h[0]))]

#     #
#     # def update_reachable_from_node(node):
#     #     if node not in reachable_from_node:
#     #         reachable_from_node[node] = set()
#     #     stack = [node]
#     #     while stack:
#     #         current = stack.pop()
#     #         if current in dict_reach_h:
#     #             for neighbor in dict_reach_h[current]:
#     #                 reachable_from_node[node].add(neighbor)
#     #                 if neighbor not in reachable_from_node[node]:
#     #                     stack.append(neighbor)
#     #                 # Also update reachable nodes for each neighbor's previously reachable nodes
#     #                 if neighbor in reachable_from_node:
#     # 
#     #       
#     #               reachable_from_node[node].update(reachable_from_node[neighbor])
#     #current source : reachable_nodes (ex: 0: [3,2] ; 1: [])
#     all_reachable = {} 


#     while dfs_stack:# ex: [0,1,2,3]
#         #TODO: add inner loop executing a single iteration of dfs on current source 
        
#         current_source = dfs_stack.pop()
#         #TODO: Flatten dictionary (ex. DFS_STACK: [0,1,2,3] ; all_reachable_flat(set): [0,2,3], current_source: 1; all_reachable_flat: [0,1,2,3], current_source: 2, etc;)
#         current_stack = dfs_stack
#         hints.append(f"Instruction: Please List all known reachable nodes.")
#         hints.append(f"Current Stack: {current_stack}")
#         hints.append(f"Pop: {current_source}")

        

#         if current_source not in dict_reach_h or len(dict_reach_h[current_source]) == 0:
#             hints.append(f"Source {current_source} has no connections, continue to next stack element.")
#             continue

#         # Ensure we visit all connections from the current source
#         neighbors = sorted(list(dict_reach_h[current_source]))
#         hints.append(f"Neighborhood of {current_source}: {neighbors}")

#         # Update reachable nodes for the current source
#         # update_reachable_from_node(current_source)

#         # Include the current source in reachable nodes
#         reachable_from_node[current_source].add(current_source)
#         hints.append(f"Reachable nodes from {current_source}: {sorted(reachable_from_node[current_source])}")

#         for node_idx in neighbors:
#             if node_idx not in visited_:
#                 dfs_stack.append(node_idx)
#                 visited_.add(node_idx)

   

#     reachable_nodes = sorted(all_reachable)
#     hints.append("List all known reachable nodes.")
#     #for loop here(ex:reachable from 0: ...,)
#     hints.append(f"Reachable Nodes: {reachable_nodes}")
# #for loop (ex: reachable from 0: )
#     return hints

#best working version- just missing the known reachable
# def _dfs_translate_reach_pred_h(neg_edges, edgelist_lookup, list_pred_h, list_color_h, list_discovery_h, 
#                                 list_final_h, list_s_prev_h, list_s_h, 
#                                 list_source_h, list_target_h, list_s_last_h, list_time):
#     visited_ = set()
#     reachable_from_node = {}  # Dictionary to keep track of reachable nodes from each node
#     all_reachable_flat = set()  # Set to keep track of all reached nodes as tuples

#     # Convert edgelist_lookup to bidirectional adjacency dictionary
#     def edgelist_to_dict(edgelist_lookup):
#         adj_dict = {}
#         for (u, v) in edgelist_lookup:
#             if u not in adj_dict:
#                 adj_dict[u] = set()
#             if v not in adj_dict:
#                 adj_dict[v] = set()
#             adj_dict[u].add(v)
#             adj_dict[v].add(u)
#         return adj_dict
    
#     adj_dict = edgelist_to_dict(edgelist_lookup)

#     # Ensure all nodes from list_s_h are included in adj_dict
#     for source_nodes in list_s_h:
#         if isinstance(source_nodes, np.ndarray) or isinstance(source_nodes, list):
#             for node_idx, source in enumerate(source_nodes):
#                 if node_idx not in adj_dict:
#                     adj_dict[node_idx] = set()
#         else:
#             raise ValueError(f"Unexpected structure in list_s_h: {source_nodes}")

#     # Initialize reachable_from_node for all nodes in adj_dict
#     for node in adj_dict:
#         reachable_from_node[node] = set()

#     # Initialize the stack with all nodes in numerical order
#     all_nodes = sorted(adj_dict.keys())
#     reach_h_stack = set()
#     for node in all_nodes:
#         reach_h_stack.add(node)
#         reachable_from_node[node] = {node}

#     # Flatten reach_h_stack for use in all_reachable_flat
    

#     hints = []
#     dfs_stack = list(all_nodes)  # Start with all nodes in numerical order

#     def find_reachable_nodes(node, adj_dict, reachable_set):
#         if node in adj_dict:
#             for neighbor in adj_dict[node]:
#                 if neighbor not in reachable_set:
#                     reachable_set.add(neighbor)
#                     find_reachable_nodes(neighbor, adj_dict, reachable_set)

#     while dfs_stack:
#         current_source = dfs_stack.pop()
#         current_stack = list(dfs_stack)
#         hints.append(f"Instruction: Please List all known reachable nodes.")
#         hints.append(f"Pop: {current_source}")
#         hints.append(f"Current Stack: {current_stack}")

#         if current_source in all_reachable_flat:
#             hints.append(f"Source {current_source} has been reached already, continue to next stack element.")
#             continue

#         all_reachable_flat.add(current_source)


#         #case for solo nodes or nodes not in adjacency dictionary
#         if current_source not in adj_dict or len(adj_dict[current_source]) == 0:
#             hints.append(f"Source {current_source} has no connections, continue to next stack element.")
#             hints.append(f"Direct neighbors for node {current_source}: []")
#             hints.append(f"Reachable nodes for node {current_source}: [{current_source}]")
#             continue

#         neighbors = sorted(list(adj_dict[current_source]))

#         # Include the current source in reachable nodes
#         reachable_from_node[current_source].add(current_source)

#         for node_idx in neighbors:
#             if node_idx not in visited_:
#                 visited_.add(node_idx)

#         # Add current source to the visited set
#         visited_.add(current_source)

#         # Add the current source and its reachable nodes to all_reachable_flat
#         all_reachable_flat.update(reachable_from_node[current_source] | {current_source})

#         # Update reachable_from_node for the current source
#         reachable_set = set()
#         find_reachable_nodes(current_source, adj_dict, reachable_set)
#         reachable_from_node[current_source].update(reachable_set)

#         # Add current reachable nodes and direct neighbors to hints
#         hints.append(f"Direct neighbors for node {current_source}: {sorted(list(adj_dict[current_source]))}")
#         hints.append(f"Reachable nodes for node {current_source}: {sorted(reachable_from_node[current_source])}")
#         # hints.append("\n")
#         # # Debugging statement to show the visited set and dfs_stack
#         # print(f"Visited nodes: {visited_}")
#         # print(f"DFS stack: {dfs_stack}")

#     # Final output for reachable nodes
#     all_reachable = set()
#     for node in reach_h_stack:
#         if node in reachable_from_node:
#             all_reachable.update(reachable_from_node[node])

#     reachable_nodes = sorted(all_reachable)
#     hints.append("")
#     hints.append("List all known reachable nodes.")
#     hints.append(f"Response: Reachable Nodes: {reachable_nodes}")

#     # Debugging statement to show the final reachable nodes
#     print(f"Final reachable nodes: {reachable_nodes}")

#     return hints

def _dfs_translate_reach_pred_h(neg_edges, edgelist_lookup, list_pred_h, list_color_h, list_discovery_h, 
                                list_final_h, list_s_prev_h, list_s_h, 
                                list_source_h, list_target_h, list_s_last_h, list_time):
    visited_ = set()
    reachable_from_node = {}  # Dictionary to keep track of reachable nodes from each node
    all_reachable_flat = set()  # Set to keep track of all reached nodes as tuples
    all_known_reachable = set()  # Set to accumulate all known reachable nodes

    # Convert edgelist_lookup to bidirectional adjacency dictionary
    def edgelist_to_dict(edgelist_lookup):
        adj_dict = {}
        for (u, v) in edgelist_lookup:
            if u not in adj_dict:
                adj_dict[u] = set()
            if v not in adj_dict:
                adj_dict[v] = set()
            adj_dict[u].add(v)
            adj_dict[v].add(u)
        return adj_dict
    
    adj_dict = edgelist_to_dict(edgelist_lookup)

    # Ensure all nodes from list_s_h are included in adj_dict
    for source_nodes in list_s_h:
        if isinstance(source_nodes, np.ndarray) or isinstance(source_nodes, list):
            for node_idx, source in enumerate(source_nodes):
                if node_idx not in adj_dict:
                    adj_dict[node_idx] = set()
        else:
            raise ValueError(f"Unexpected structure in list_s_h: {source_nodes}")

    # Initialize reachable_from_node for all nodes in adj_dict
    for node in adj_dict:
        reachable_from_node[node] = set()

    # Initialize the stack with all nodes in numerical order
    all_nodes = sorted(adj_dict.keys())
    reach_h_stack = set()
    for node in all_nodes:
        reach_h_stack.add(node)
        reachable_from_node[node] = {node}

    # Flatten reach_h_stack for use in all_reachable_flat
    all_reachable_flat.update(reach_h_stack)

    hints = []
    dfs_stack = list(all_nodes)  # Start with all nodes in numerical order

    def find_reachable_nodes(node, adj_dict, reachable_set):
        if node in adj_dict:
            for neighbor in adj_dict[node]:
                if neighbor not in reachable_set:
                    reachable_set.add(neighbor)
                    find_reachable_nodes(neighbor, adj_dict, reachable_set)

    while dfs_stack:
        current_source = dfs_stack.pop()
        current_stack = list(dfs_stack)
        hints.append(f"Instruction: Please List all known reachable nodes.")
        hints.append(f"Pop: {current_source}")
        hints.append(f"Current Stack: {current_stack}")

        if current_source in all_reachable_flat:
            hints.append(f"Source {current_source} has been reached already, continue to next stack element.")
            continue

        all_reachable_flat.add(current_source)

        if current_source not in adj_dict or len(adj_dict[current_source]) == 0:
            hints.append(f"Source {current_source} has no connections, continue to next stack element.")
            hints.append(f"Direct neighbors for node {current_source}: []")
            hints.append(f"Reachable nodes for node {current_source}: [{current_source}]")
            all_known_reachable.update({current_source})
            hints.append(f"All known reachable: {sorted(all_known_reachable)}")
            continue

        # Ensure we visit all connections from the current source
        neighbors = sorted(list(adj_dict[current_source]))
        hints.append(f"Neighborhood of {current_source}: {neighbors}")

        # Include the current source in reachable nodes
        reachable_from_node[current_source].add(current_source)

        for node_idx in neighbors:
            if node_idx not in visited_:
                dfs_stack.append(node_idx)
                visited_.add(node_idx)

        # Add current source to the visited set
        visited_.add(current_source)

        # Update reachable_from_node for the current source by calling find_reachable_nodes
        reachable_set = set()
        find_reachable_nodes(current_source, adj_dict, reachable_set)
        reachable_from_node[current_source].update(reachable_set)

        # Add the current source and its reachable nodes to all_reachable_flat
        all_reachable_flat.update(reachable_from_node[current_source] | {current_source})

        # Update all_known_reachable
        all_known_reachable.update(reachable_from_node[current_source])

        # Add current reachable nodes and direct neighbors to hints
        hints.append(f"Direct neighbors for node {current_source}: {sorted(list(adj_dict[current_source]))}")
        hints.append(f"Reachable nodes for node {current_source}: {sorted(reachable_from_node[current_source])}")
        hints.append(f"All known reachable: {sorted(all_known_reachable)}")

        # Debugging statement to show the visited set and dfs_stack
        print(f"Visited nodes: {visited_}")
        print(f"DFS stack: {dfs_stack}")

    # Final output for reachable nodes
    all_reachable = set()
    for node in reach_h_stack:
        if node in reachable_from_node:
            all_reachable.update(reachable_from_node[node])

    reachable_nodes = sorted(all_reachable)
    hints.append("")
    hints.append("List all known reachable nodes.")
    hints.append(f"Response: Reachable Nodes: {reachable_nodes}")

    # Debugging statement to show the final reachable nodes
    print(f"Final reachable nodes: {reachable_nodes}")

    return hints


def _dfs_translate_output(list_pred):
    list_out_idxs = [str(node_idx) for node_idx, pred_idx in enumerate(list_pred) if pred_idx != node_idx]
    return f"### Reachable Nodes: {', '.join(list_out_idxs)}"# if len(list_out_idxs) > 0 else "There are no reachable nodes"

def _datapoint_to_dict(dp):
    return {"name":dp.name,
            "location":dp.location,
            "data":dp.data}


# 
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


def convert_3d_array_to_2d_list(array_3d): #helper function for color field 
    """
    Convert a 3D numpy array to a 2D list where each row of each 2D slice is treated as a separate entity.
    """
    return [row.tolist() for slice_2d in array_3d for row in slice_2d]


def translate_hints(alg, neg_edges, edgelist_lookup, hints):
    hints_dict = _datapoints_list_to_dict(hints)
    


    # inputs_dict = edgelist_lookup
    # list_edge = inputs_dict.get("list_edge")
    print(edgelist_lookup)

    if alg in ["bfs"]:
        # unweighted graph algorithms
        list_reach_h = _preprocess_hint_matrix(alg, hints_dict["reach_h"]["data"])
        list_pred_h = _preprocess_hint_matrix(alg, hints_dict["pi_h"]["data"])
        list_h = _bfs_translate_reach_pred_h(neg_edges, edgelist_lookup, list_reach_h, list_pred_h)
        return list_h
    elif alg in ["dfs"]:
        # unweighted graph algorithms
        # list_edge = hints_dict["list_edge"]
        list_pred_h = _preprocess_hint_matrix(alg, hints_dict["pi_h"]["data"])
        list_color_h = _preprocess_hint_matrix(alg, hints_dict["color"]["data"])
        list_discovery_h = _preprocess_hint_matrix(alg, hints_dict["d"]["data"])
        list_final_h = _preprocess_hint_matrix(alg, hints_dict["f"]["data"])
        list_s_prev_h = _preprocess_hint_matrix(alg, hints_dict["s_prev"]["data"])
        list_s_h = _preprocess_hint_matrix(alg, hints_dict["s"]["data"])
        list_source_h = _preprocess_hint_matrix(alg, hints_dict["u"]["data"])
        list_target_h = _preprocess_hint_matrix(alg, hints_dict["v"]["data"])
        list_s_last_h = _preprocess_hint_matrix(alg, hints_dict["s_last"]["data"])
        list_time = _preprocess_hint_matrix(alg, hints_dict["time"]["data"])
        list_h = _dfs_translate_reach_pred_h( neg_edges, edgelist_lookup, list_pred_h, list_color_h, list_discovery_h, 
        list_final_h, list_s_prev_h, list_s_h, 
        list_source_h, list_target_h, list_s_last_h, list_time)
        return list_h
    elif alg in ["dka", "bfd"]:
        #potentially weighted graph algorithms
        raise NotImplementedError(f"[WILL BE REPLACED] No hint translation functionality has been implemented for {alg}")
    else:
        raise NotImplementedError(f"No hint translation functionality has been implemented for {alg}")


def _translate_inputs(alg, inputs):
    inputs_dict = _datapoints_list_to_dict(inputs)

    if alg in ["bfs"]:
        # unweighted graph algorithms
        algorithm = alg
        list_edge = _translate_unweighted_graph(inputs_dict["adj"]["data"])
        source = _translate_source_node(inputs_dict["s"]["data"])
        return algorithm, list_edge, source
    elif alg in ["dfs"]:
        algorithm = alg
        list_edge = _translate_unweighted_graph(inputs_dict["adj"]["data"])
        return algorithm, list_edge
    elif alg in ["dka", "bfd"]:
        #potentially weighted graph algorithms
        raise NotImplementedError(f"[WILL BE REPLACED] No input translation functionality has been implemented for {alg}")
    else:
        raise NotImplementedError(f"No input translation functionality has been implemented for {alg}")

def hash_edgelist(edgelist):
    canonicalEdges = sorted([str(sorted(edge)) for edge in edgelist])  # Canonical form and sort
    return hash(",".join(canonicalEdges))  # Convert to unique representation

def sample_data(args):
    clrs_training_data = {}
    clrs_validation_data = {}
    clrs_testing_data = {}
    
    trans_training_data = {}
    trans_validation_data = {}
    trans_testing_data = {}
    
    graph_sizes =  [4] #range(3, args.graph_sizes + 1)
    
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
            print(train_sample)
            print( _datapoints_list_to_dict(train_sample.features.inputs))
            print( _datapoints_list_to_dict(train_sample.features.hints))
            print( _datapoints_list_to_dict(train_sample.outputs))
            
            inputs = _translate_inputs(args.algorithm, train_sample.features.inputs)
             
            edgelist_hash = hash_edgelist(inputs[1])
            if edgelist_hash in unique_graphs:
                continue
            hints = translate_hints(args.algorithm, args.neg_edges, set(inputs[1]), train_sample.features.hints)
            # Add print statements to display the DFS hints
            print("DFS Hints:")
            for hint in hints:
                print(hint)
            return;   
            
            # hints = translate_hints(args.algorithm, args.neg_edges, set(inputs[0]), train_sample.features.hints)
            outputs = translate_outputs(args.algorithm, train_sample.outputs)
            # return
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
            
            hints = translate_hints(args.algorithm, args.neg_edges, set(inputs[0]), test_sample.features.hints)
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
    
def main():
    args = data_utils.parse_args()
    sample_data(args)
    
if __name__ == "__main__":
    main()
