import pdb
from tqdm import tqdm
import numpy as np
import torch
def aggregate_user_embedding(node_influence, data_path):
    user_emb = torch.from_numpy(np.loadtxt(data_path, dtype=np.float32))
    print('aggregate user embedding to get group embedding')
    for index in tqdm(range(0, len(user_emb))):
        user_emb[index] *= float(node_influence[index][1])
    group_emb = torch.sum(user_emb, dim=0)
    group_emb = group_emb.reshape(1, len(group_emb))
    return group_emb
