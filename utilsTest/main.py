#!/usr/bin/env python36
# -*- coding: utf-8 -*-
import pdb
import time
import pickle
import argparse
import traceback
from tqdm import tqdm
import torch.nn.functional as F
from model.Yelp.SRGNN.model import *
from collections import defaultdict
from model.Yelp.SRGNN.utils import build_graph, Data, split_validation

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Yelp', help='dataset name:Yelp')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=9, help='the original number is 100,hidden state size')
parser.add_argument('--epoch', type=int, default= 20, help='30, the original number is 30, the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
opt = parser.parse_args()

def get_poi_emb(model, poi_id):
    poi_id_data = [[poi_id]]
    target_data = [[poi_id]]
    
    data = (poi_id_data, target_data)
    data = Data(data, shuffle=True)

    slices = data.generate_batch(1)
    target, score = forward(model, slices, data)

    return score

def main():
    ground_truth = defaultdict(list)
    truth_data = open('./data/Yelp/Yelp_check_ins.txt', 'r').readlines()
    for eachline in truth_data:
        uid, lid, _ = eachline.strip().split()
        uid, lid = int(uid), int(lid)
        ground_truth[uid].append(lid)
    # 将签到数据拿出来
    user_poi_checkIn_data = list(ground_truth.values())
    # 把每一个用户的poi签到序列的最后一项提取出来作为整个签到序列的标签/预测值/y
    target_data = []
    for poi_sequence in user_poi_checkIn_data:
        target_data.append(poi_sequence[-1])
        del poi_sequence[-1]  # 再把最后一个元素在原序列中删除掉
    train_data = (user_poi_checkIn_data, target_data)    
    test_data = (train_data[0][2451:-1], train_data[1][2451:-1]) 
    train_data = (train_data[0][0:2451], train_data[1][0:2451])
    
    train_data = Data(train_data, shuffle=True)
    test_data = Data(test_data, shuffle=False)

    # del all_train_seq, g
    if opt.dataset == 'Yelp':
        n_node = 18995
     

    model = trans_to_cuda(SessionGraph(opt, n_node))

    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0
    for epoch in tqdm(range(0, opt.epoch)):
        #print('-------------------------------------------------------')
        #print('epoch: ', epoch)
        model, hit, mrr = train_test(model, train_data, test_data)
        flag = 0
        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1
        #print('Best Result:')
        #print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d'% (best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    #print('-------------------------------------------------------')
    end = time.time()
    #print("Run time: %f s" % (end - start))
    return model
