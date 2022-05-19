import os
import numpy as np
from collections import defaultdict

def read_context_features(path):
    #读取context_features数据
    print('read_context_features data...')
    context_features = np.loadtxt(path, dtype=np.int32)
    print('succeed...')
    return context_features

def read_poi_context_intecrection(path):
    print('read_poi_context_intecrection data...')
    # 读取每一个poi对应的context_id并返回
    ground_truth = defaultdict(set)
    truth_data = open(path, 'r').readlines()
    for eachline in truth_data:
        user_id, poi_id, frequnecy, conte_id = eachline.strip().split()
        user_id, poi_id, frequnecy, conte_id = int(user_id), int(poi_id), int(frequnecy), int(conte_id)
        ground_truth[poi_id].add(conte_id)
    print('succeed...')
    return ground_truth


def read_ground_truth(test_file):
    ground_truth = defaultdict(set)
    truth_data = open(test_file, 'r').readlines()
    for eachline in truth_data:
        uid, lid, _ = eachline.strip().split()
        uid, lid = int(uid), int(lid)
        ground_truth[uid].add(lid)
    return ground_truth

