import pdb
import numpy as np
import networkx as nx
from tqdm import tqdm

def create_graph(data_size_path, social_relations_path):
    # 根据传入的数据路径，读取文件数据，构建graph 
    data_size = np.loadtxt(data_size_path, dtype=np.int32)
    all_memners = set(range(data_size[0]))
    
    G = nx.Graph()
    G.add_nodes_from(all_memners)

    social_relations = np.loadtxt(social_relations_path, dtype=np.int32)
    print('create social network graph...')
    for relation in tqdm(social_relations):
        G.add_edge(relation[0], relation[1])
    
    # remove all isolates in the graph

    return G

def create_tripartite_graph(social_relations):
    # 根据传入的数据路径，读取文件数据，构建tripartite graph 
    all_members = set(range(len(social_relations)))
    
    G = nx.Graph()
    G.add_nodes_from(all_members)

    print('create social network graph...')
    for relation in tqdm(social_relations):
        G.add_edge(relation[0], relation[1])
    
    # remove all isolates in the graph
    return G



