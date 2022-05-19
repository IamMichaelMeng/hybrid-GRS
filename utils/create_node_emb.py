import torch
import torch.nn as nn
import numpy as np
import pdb
def create_static_user_emb(num_nodes, embedding_dim, user_info):
    emb = None
    emb = nn.Embedding(num_nodes, embedding_dim)
    emb.weight.data = torch.tensor(user_info)
    return emb

def create_user_emb(num_nodes, embedding_dim):
    # DO NOT CHANGE THIS DATA
    torch.manual_seed(1)

    emb = None
    emb = nn.Embedding(num_nodes, embedding_dim)
    emb.weight.data = torch.rand(num_nodes, embedding_dim)
    return emb

