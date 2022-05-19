'''
分别使用欧几里得、余弦相似度、皮尔森系数三种方法
寻找虚拟用户的相似用户，并针对三种类型的相似用户组分别生成三个group-poi
数据集
'''
# euclidean distance, cosine similarity, adjusted cosine similarity, Pearson coefficient
import pdb
import numpy as np
from tqdm import tqdm
from numpy.linalg import norm
from scipy.stats import pearsonr
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform

def takeSecond(elem):
    return elem[1]


def create_nagative_interection(positive_interection, poi_num, dataset_name):
    file = open('../datasets/group/group_poi/Yelp_'+ dataset_name +'_negative_group_poi.txt', mode='a')
    group_id = -1
    for poi in tqdm(range(0, poi_num)):
        if not(poi in positive_interection):
            file.writelines([str(group_id)+' ', str(poi)+' ', str(-1), '\n'])
    file.close()


def create_negative_group_poi_interaction(dataset_name, positive_interection):
    if dataset_name == 'Yelp':
        data_size = np.loadtxt('../datasets/Yelp/Yelp_data_size.txt', dtype=np.int32)
        poi_num = data_size[1]
        create_nagative_interection(positive_interection, poi_num, dataset_name)

def get_common_user_poi_interection(user_id):
    groud_truth = set() 
    truth_data = open('../datasets/Yelp/Yelp_check_ins.txt', 'r').readlines()
    for eachline in truth_data:
        uid, poi_id, _ = eachline.strip().split()
        uid, poi_id = int(uid), int(poi_id)
        if (uid == user_id):
            groud_truth.add(poi_id)

    return groud_truth

# 根据输入的数据集，生成对应数据集的group_poi数据集
def create_single_positive_group_poi_interaction(user_ids, data_name):
    # get user-poi interection 
    common_user_poi_interection_set = set() 
    for user_id in tqdm(user_ids):
        interection_data = get_common_user_poi_interection(user_id)
        common_user_poi_interection_set.update(interection_data)
    file = open('../datasets/group/group_poi/Yelp_'+ data_name +'_single_positive_group_poi.txt', mode='a')
    group_id = -1 # set group_id = -1
    for elem in common_user_poi_interection_set:
        file.writelines([str(group_id)+' ', str(elem)+' ', str(1), '\n'])
    file.close()

    return common_user_poi_interection_set






# create postive group-poi interaction data
def create_positive_group_poi_interaction(euclidean_dis_set, cosine_similarity_set, person_coefficient_set, data_name):
    common_user_id = euclidean_dis_set.intersection(cosine_similarity_set, person_coefficient_set)
    common_user_poi_interection_set = set() 

    # get user-poi interection 
    for user_id in tqdm(common_user_id):
        interection_data = get_common_user_poi_interection(user_id)
        common_user_poi_interection_set.update(interection_data)
    file = open('../datasets/group/group_poi/Yelp_'+ data_name +'_positive_group_poi.txt', mode='a')
    group_id = -1 # set group_id = -1
    for elem in common_user_poi_interection_set:
        file.writelines([str(group_id)+' ', str(elem)+' ', str(1), '\n'])
    file.close()

    return common_user_poi_interection_set

# euclidean distance 
def euclidean_distance():
    euclidean_dis_list = []
    euclidean_dis_set = set()
    for user_idx, user_info in enumerate(tqdm(user_features)):
        dist = 1. / (1 + np.linalg.norm(group_emb - user_info)) # map to [0, 1]
        euclidean_dis_list.append([user_idx, dist])

    # sorted by dist in decreasing order
    euclidean_dis_list.sort(key=takeSecond, reverse=True) # 按照距离降序排序
    euclidean_dis_list = np.array(euclidean_dis_list)[:100] # 只取前100
    euclidean_dis_set = euclidean_dis_list[:, 0]  # 取前100个用户对应的id
    # 将结果保存起来
    np.savetxt('../datasets/group/group_poi/Yelp_euclidean_dis_set.txt', euclidean_dis_set, fmt='%d')
    return euclidean_dis_set.astype(int)


def cosine_similarity():
    cosine_similarity_list = []
    cosine_similarity_set = set()
    for user_idx, user_info in enumerate(tqdm(user_features)):
        num = float(np.dot(user_info, group_emb))  # dot
        denom = np.linalg.norm(user_info) * np.linalg.norm(group_emb)  # multiplile
        similarity = 0.5 + 0.5 * (num / denom) if denom != 0 else 0

        cosine_similarity_list.append([user_idx, similarity])
    
    cosine_similarity_list.sort(key=takeSecond, reverse=True)
    cosine_similarity_list = np.array(cosine_similarity_list)[:100]
    cosine_similarity_set = cosine_similarity_list[:,0]
    # 将结果保存起来
    np.savetxt('../datasets/group/group_poi/Yelp_cosine_dis_set.txt', cosine_similarity_set, fmt='%d')
    return cosine_similarity_set.astype(int)


def person_coefficient():
    person_coefficient_list = []
    person_coefficient_set = set()
    for user_idx, user_info in enumerate(tqdm(user_features)):
        pccs = pearsonr(group_emb, user_info)
        #pccs = np.corrcoef(group_emb, user_info[1:])
        person_coefficient_list.append([user_idx, pccs[0]])

    person_coefficient_list.sort(key=takeSecond, reverse=True)
    person_coefficient_list = np.array(person_coefficient_list)[:100]
    person_coefficient_set = person_coefficient_list[:,0]
    # 将结果保存起来
    np.savetxt('../datasets/group/group_poi/Yelp_person_dis_set.txt', person_coefficient_set, fmt='%d')
    return person_coefficient_set.astype(int)


if __name__ =='__main__':
    user_features_file_path = []
    group_emb = np.loadtxt('../datasets//group/Yelp_group_emb.txt', dtype=np.float32)
    group_emb = group_emb/np.linalg.norm(group_emb)  # 归一化
    user_features_file_path.append('../datasets/Yelp/Yelp_user_features.txt')

    user_features = np.loadtxt(user_features_file_path[0], dtype=np.int32)
    user_features = user_features.reshape(user_features.shape[0], user_features.shape[1])

    dataset_name = 'Yelp' 
    
    # all kinds of similarity function
    euclidean_dis_set = set(euclidean_distance())
    cosine_similarity_set = set(cosine_similarity())
    person_coefficient_set = set(person_coefficient())
    print('开始生成group-poi签到数据')    
    create_single_positive_group_poi_interaction(euclidean_dis_set, 'enclidean')
    create_single_positive_group_poi_interaction(cosine_similarity_set, 'cosine')
    create_single_positive_group_poi_interaction(person_coefficient_set, 'person')


    
