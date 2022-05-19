import pdb
import numpy as np
from tqdm import tqdm
from collections import defaultdict
def create_new_check_in():
    #构建携带weather的新的check_in数据集
    check_in_data = np.loadtxt('../../../datasets/Foursquare/Foursquare_check_ins_tripartite_graph.txt', dtype=np.int32).tolist()
    weather_data = np.loadtxt('../../../datasets/weather_data.txt', dtype=np.int32).tolist()
    context_features = []
    for index, check_in in enumerate(tqdm(check_in_data)):
        index -= 1
        temp = [check_in[3], check_in[2]]
        temp.extend(weather_data[index][1:10])
        context_features.append(temp)
    np.savetxt('../../../datasets/Foursquare/Foursquare_context_features.txt', context_features, fmt='%d', delimiter=' ')
def read_ground_truth(path):
    ground_truth = []
    truth_data = open(path, 'r').readlines()
    for eachline in truth_data:
        poi_id, latitude, longitude = eachline.strip().split()
        poi_id = int(poi_id)
        latitude, longitude = float(latitude), float(longitude)
        ground_truth.append([poi_id, latitude, longitude])
    return ground_truth

def data_masks():
    # 构建poi_features数据，将数据纬度拉到11维
    # 读取数据
    poi_categories = np.loadtxt('../../../datasets/Foursquare/Foursquare_poi_categories.txt', dtype=np.int32).tolist()
    poi_coos = read_ground_truth('../../../datasets/Foursquare/Foursquare_poi_coos.txt')

    # 给数据进行mask操作，让其dimension变成11
    mask_length = 11
    item_tail = [0]
    file = open('../../../datasets/Foursquare/Foursquare_poi_features.txt', mode='a')

    for poi in tqdm(poi_categories):
        try:
            temp = [str(poi[0])+' ',str(poi[1])+' ', str(poi_coos[poi[0]][1])+' ', str(poi_coos[poi[0]][2])+' ', str('0 0 0 0 0 0 0')+' ', '\n']
            file.writelines(temp)
        except:
            continue
    file.close()
    '''
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_msks, len_max
    '''
if __name__=='__main__':
    create_new_check_in()
    #data_masks()
