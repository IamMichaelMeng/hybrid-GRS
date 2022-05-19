# 计算模型的准确度
import pdb
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import datetime
import time
curr_time = datetime.datetime.now()


def F1_score(precision, recall):
    if (precision+recall == 0):
        return 0.0
    else:
        return (1.0 * 2 * precision * recall) / (precision+recall)

def precision(prediction, actual):
     return 1.0 * len(set(actual) & set(prediction)) / len(prediction)

def recall(prediction, actual):
    return 1.0 * len(set(actual) & set(prediction)) / len(actual)

def mean_average_precision_k(prediction, actual):
    hits = 0
    sum_precs = 0
    for n in range(len(prediction)):
        if prediction[n] in actual:
            hits += 1
            sum_precs += hits / (n + 1.0)
    if hits > 0:
        return sum_precs / len(actual)
    else:
        return 0

def takeSecond(elem):
    return elem[1]


def calculate_performance(prediction_data, actual, similarity_cate):
    sample_k = [5,10,15,20]
    #将预测数据按照预测评分降序排序
    prediction_data.sort(key=takeSecond, reverse=True)
    # 提取预测poi_id
    prediction_poi_ids = np.array(prediction_data)[:,0]
    p = precision(prediction_poi_ids, actual)
    r = recall(prediction_poi_ids, actual)
    m = mean_average_precision_k(prediction_poi_ids, actual)
    F = F1_score(p, r)
    new_performance_result = defaultdict(list)
    for k in tqdm(sample_k):
        prediction_k_ids = prediction_poi_ids[0:k]
        Precision = precision(prediction_k_ids, actual)
        Recall = recall(prediction_k_ids, actual) 
        Map = mean_average_precision_k(prediction_k_ids, actual)
        F1Score = F1_score(Precision, Recall)
        new_performance_result[k].append(Precision)
        new_performance_result[k].append(Recall)
        new_performance_result[k].append(Map)
        new_performance_result[k].append(F1Score)
    # 下面把结果保存到文件中
    print('将结果写入文件....')
    timestamp=datetime.datetime.strftime(curr_time,'%Y-%m-%d-%H-%M-%S')
    file = open('../metric/metric/Yelp_result_'+timestamp+'.txt', mode = 'a')
    temp = [str(p)+' ', str(r)+' ', str(m)+' ', str(F), '\n']
    file.writelines(temp)
    

    file.write(similarity_cate+'\n')
    for index, data in enumerate(new_performance_result.values()):
        file.write('new_performance_result['+str(sample_k[index])+']'+'\n')
        temp = [str(data[0])+' ', str(data[1])+' ', str(data[2])+' ',str(data[3]) ,'\n']
        file.writelines(temp)
    file.close()

def read_prediction_data():
    data_list = []
    data = open('../results/Yelp_result_2022-05-15-11-26-57.txt', 'r').readlines()
    for eachline in tqdm(data):
        poi_id, score = eachline.strip().split()
        poi_id, score = int(poi_id), float(score)
        data_list.append([poi_id, score])
    return data_list



def read_ground_truth(data_path):
    ground_truth = [] 
    truth_data = open(data_path, 'r').readlines()
    for eachline in truth_data:
        _, lid,_ = eachline.strip().split()
        ground_truth.append(int(lid))
    return ground_truth



if __name__=='__main__':
    prediction_data = read_prediction_data()
    cate = ['encidean', 'cosine', 'person']
    data_path = [
    '../datasets/group/group_poi/Yelp_enclidean_single_positive_group_poi.txt',
    '../datasets/group/group_poi/Yelp_cosine_single_positive_group_poi.txt',
    '../datasets/group/group_poi/Yelp_person_single_positive_group_poi.txt']
    # 计算准确度、召回度以及平均损失
    for i in range(3):
        real_data = read_ground_truth(data_path[i])
        calculate_performance(prediction_data, real_data, cate[i])
