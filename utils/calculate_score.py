# 计算group-poi的最终预测评分
import pdb
from sklearn.preprocessing import normalize
import torch
def calculate_final_score_without_context(group_emb, SBGNN_poi_emb, SRGNN_poi_emb):
    randindex = torch.randperm(SBGNN_poi_emb.shape[1])

    SRGNN_poi_emb = SRGNN_poi_emb[0][randindex].reshape(1, SBGNN_poi_emb.shape[1])
    group_emb = group_emb[randindex].reshape(1, SBGNN_poi_emb.shape[1])
    group_emb = normalize(group_emb, axis=0, norm='max') # 归一化处理
    poi_emb = SBGNN_poi_emb + SRGNN_poi_emb

    # 计算group-poi预测分数
    score = float(torch.sigmoid(torch.mm(torch.from_numpy(group_emb), poi_emb.T)))
    return score



def calculate_final_score(group_emb, SBGNN_poi_emb, SRGNN_poi_emb, context_feature):
    pdb.set_trace()
    randindex = torch.randperm(SBGNN_poi_emb.shape[1])

    SRGNN_poi_emb = SRGNN_poi_emb[0][randindex].reshape(1, SBGNN_poi_emb.shape[1])
    group_emb = group_emb[randindex].reshape(1, SBGNN_poi_emb.shape[1])
    group_emb = normalize(group_emb, axis=0, norm='max') # 归一化处理
    poi_emb = SBGNN_poi_emb + SRGNN_poi_emb

    # 计算group-poi预测分数
    score = float(torch.sigmoid(torch.mm(torch.from_numpy(group_emb), poi_emb.T)))
    return score

