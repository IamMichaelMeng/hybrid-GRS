from sklearn.preprocessing import normalize
import sys
sys.path.append('../../')
import pdb
import torch
from torch import nn as nn
from scipy.sparse import coo_matrix
import pandas as pd
import numpy as np
from tqdm import tqdm
from numpy import diag
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.optim import Adam
from torch.nn import MSELoss
from utils.GCFmodel import GCF 
from collections import defaultdict
from utils.NGCF_utils import  read_checkin_data, read_checkin_data_include_context, Foursquare
from os import path


def as_num(x):
    y = '{:.10f}'.format(x)  # .10f 保留10位小数
    return y



def main():
    #读取数据并坐归一化处理
    poi_features = normalize(np.loadtxt('./datasets/Foursquare/Foursquare_poi_features.txt', dtype=np.float32), axis=0, norm='max')
    context_features = normalize(np.loadtxt('./datasets/Foursquare/Foursquare_context_features.txt', dtype=np.float32), axis=0, norm='max')
    data_size = np.loadtxt('./datasets/Foursquare/Foursquare_data_size.txt', dtype=np.int32)
    contextNum = int(124933) # 获取num_context，其实和poi相等，因为二者是一对一的
    poiNum = int(data_size[1]) # 获取num_poi
    
    # default epoch is 60
    para = {'epoch':1, 'lr':0.01, 'batch_size':2048, 'train':0.8}
    rt = read_checkin_data_include_context('./datasets/Foursquare/Foursquare_check_ins_tripartite_graph_new.txt') # 读取数据 
    ds = Foursquare(rt) 
    # 训练、测试数据集分割
    trainLen = int(para['train']*len(ds)) 
    train,test = random_split(ds,[trainLen,len(ds)-trainLen])
 
    dl = DataLoader(train, batch_size=para['batch_size'],shuffle=True,pin_memory=True)
    model = GCF(poiNum,contextNum, rt, poi_features, context_features, embedSize=11, layers=[11,11,])  #embedding size is 80
    optim = Adam(model.parameters(), lr=para['lr'],weight_decay=0.001)
    lossfn = MSELoss()
    min_loss = 10000
    for i in tqdm(range(para['epoch'])):
        epoch_loss = 0
        for id,batch in enumerate(dl):
            optim.zero_grad()
            prediction = model(batch[0], batch[1]) # 一个batch是2048组数据, 每一组数据都对应一个预测值，所以prediction纬度也是2048
            loss = lossfn(batch[2].float(),prediction)
            epoch_loss += loss
            loss.backward()
            optim.step()
        if float(as_num(epoch_loss)) < min_loss:
            min_loss = epoch_loss
            torch.save(model, './model/Foursquare/model.pkl') #将模型及其参数保留下来
    # 训练结束，下面计算权值
    dl = DataLoader(ds, batch_size=int(124933))
    model = torch.load('./model/Foursquare/model.pkl')#加载最佳的模型参数
    for id, batch in enumerate(dl):
        prediction = model(batch[0], batch[1])

    return prediction
