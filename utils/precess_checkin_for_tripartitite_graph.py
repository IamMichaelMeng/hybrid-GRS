import pdb
import numpy as np
from tqdm import tqdm

def update_context_id():
    #修改context_features.txt文件，将context_id的起始下标修改为0
    context_features = np.loadtxt('../datasets/Foursquare/Foursquare_context_features.txt', dtype=np.int32)
    context_features[:,0] -= 16025 #在原基础上直接减去13474，让其下标从0开始
    np.savetxt('../datasets/Foursquare/Foursquare_context_features.txt', context_features, fmt='%d', delimiter=' ')#保存数据


def create_check_in_with_context_new():
    # 在原check_in数据集增加一列context_id
    check_in_data = np.loadtxt('../datasets/Foursquare/Foursquare_check_ins.txt', dtype=np.int32).tolist()
    check_in_num = len(check_in_data)

    index = 0  # context_id的起始下标
    for check_in in tqdm(check_in_data):
        check_in.append(index)  # 追加context_id
        index+= 1
    np.savetxt('../datasets/Foursquare/Foursquare_check_ins_tripartite_graph_new.txt', check_in_data, fmt='%d', delimiter=' ')

def create_check_in_with_context():
    # 把原check_in数据集的poi_id给改掉，然后增加一列context_id
    check_in_data = np.loadtxt('../datasets/Foursquare/Foursquare_check_ins.txt', dtype=np.int32).tolist()
    check_in_num = len(check_in_data)

    index = 16025  # context_id的起始下标
    for check_in in tqdm(check_in_data):
        check_in[1] += 2551 #修改poi_id，在原数字的基础上全部增加2551
        check_in.append(index)  # 追加context_id
        index+= 1
    np.savetxt('../datasets/Foursquare/Foursquare_check_ins_tripartite_graph.txt', check_in_data, fmt='%d', delimiter=' ')

def create_new_check_in_dataset():
    # 构建user-context、context-poi两个新的数据集
    check_in_data = np.loadtxt('../datasets/Foursquare/Foursquare_check_ins_tripartite_graph.txt', dtype=np.int32)

    # 构建两个list集合
    user_context, context_poi = [], []
    for check_in in tqdm(check_in_data):
        user_context.append([check_in[0], check_in[3]])
        context_poi.append([check_in[3], check_in[1]])
    
    np.savetxt('../datasets/Foursquare/Foursquare_user_context.txt', user_context, fmt='%d', delimiter=' ')
    np.savetxt('../datasets/Foursquare/Foursquare_context_poi.txt', context_poi, fmt='%d', delimiter=' ')

if __name__=='__main__':
   #create_check_in_with_context()
   #create_new_check_in_dataset()
   #create_check_in_with_context_new()
   update_context_id()
   



