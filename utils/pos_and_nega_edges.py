import torch
import random
import numpy as np

# sampling  negative edges
def sample_negative_edges(G, num_neg_samples):
    neg_edge_list = []

    pos_set = set(G.edges())
    visited_set = set()

    node_list = list(G.nodes())
    random.shuffle(node_list)
    
    for n_i in node_list:
        for n_j in node_list:
            if n_i == n_j \
            or (n_i,n_j) in pos_set or (n_j,n_i) in pos_set \
            or (n_i,n_j) in visited_set or (n_j, n_i) is visited_set:
                continue
                
            neg_edge_list.append((n_i,n_j))
            visited_set.add((n_i,n_j))
            visited_set.add((n_j,n_i))
            if len(neg_edge_list) == num_neg_samples:
                return neg_edge_list

def graph_to_edge_list(G):
    edge_list = []
    edge_list = list(G.edges())
    return edge_list


def edge_list_to_tensor(edge_list):
    edge_index = torch.tensor([])
    edge_index = torch.tensor(np.array(edge_list), dtype=torch.long)
    edge_index = edge_index.T
    return edge_index
