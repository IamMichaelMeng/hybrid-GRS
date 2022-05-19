import numpy as np
import networkx as nx
from tqdm import tqdm

def get_neighbors_within_R(G, n, R):
    '''
    @Description: 根据种子节点集合获取其 ``R阶之内'' 的所有邻居
    @Param: G, 种子节点n, 阶数R (R >= 0, R=0表示节点自身, R=1表示邻居)
    @Return: 
    '''
    if R == 0:
        return set([n])
    if R == 1:
        return set([n]) | set(G[n])
    else:
        return get_neighbors_within_R(G, n, R - 1) | expand_neighbors(
            G, get_neighbors_within_R(G, n, R - 1))


def get_neighbors_of_R(G, n, R):
    '''
    @Description: 根据种子节点集合获取其 ``第R阶'' 的所有邻居
    @Param: G, 种子节点n, 阶数R (R >= 0, R=0表示节点自身, R=1表示邻居)
    @Return: 
    '''
    if R == 0:
        return set([n])
    if R == 1:
        return get_neighbors_within_R(G, n, R).difference(
            get_neighbors_of_R(G, n, R - 1))
    else:
        return get_neighbors_within_R(G, n, R).difference(
            get_neighbors_within_R(G, n, R - 1))


def expand_neighbors(G, seeds):
    '''
    @Description: 根据种子节点集合获取其所有邻居
    @Param: G, 种子节点的set seeds
    @Return: 
    '''
    return set([x for n in seeds for x in G[n]])


def r_neighbors_dict(G, seed, R):
    '''
    @Description: 计算节点 seed列表 的R阶邻居
    @Param: G, seed 列表, 阶数 R
    @Return: 字典类型，key为阶数，value为对应距离的节点列表
    '''
    return {r: get_neighbors_of_R(G, seed, r) for r in range(1, R + 1)}


def cal_inf(G, u, R=1):
    '''
    @Description: 计算节点影响力
    @Param: G, 节点u, R:阶数，默认为一阶，即直接邻居
    @Return: 节点的影响力
    '''
    rnd = r_neighbors_dict(G, u, R)
    return sum([1 / (r * r * len(G[v])) for r in rnd for v in rnd[r] if G[v]])

def stable_sigmoid(x):

    sig = np.where(x < 0, np.exp(x)/(1 + np.exp(x)), 1/(1 + np.exp(-x)))
    return sig

def get_node_influence(G):
    node_influence = []
    print('calculate graph node influence...')
    for n in tqdm(G.nodes()):
        node_influence.append([n, stable_sigmoid(cal_inf(G, n))])
    return node_influence
