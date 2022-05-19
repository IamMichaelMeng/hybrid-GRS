# csv reader
import pdb
import csv
import numpy as np
import time
from tqdm import tqdm
def takeElem(elem):
    return elem[0]

if __name__ == '__main__':
    #下面开始将天气数据集合并到已有的check_in数据集中
    check_in_data = np.loadtxt('../datasets/Foursquare/Foursquare_check_ins.txt', dtype=np.int64).tolist()
    poi_category = np.loadtxt('../datasets/Foursquare/Foursquare_poi_categories.txt', dtype=np.int32).tolist()
    poi_category.sort(key=takeElem) # 将数据按照poi_id升序排序
    weather_data = np.loadtxt('../datasets/weather_data.txt', dtype=np.float32).tolist()

    new_check_in_data = []
    
    print('merge check_in_data and weather_data')
    for index, check_in in enumerate(tqdm(check_in_data)):
        check_in.append(poi_category[check_in[1]][1]) # 先把poi的种类追加到check_in数据结尾
        check_in.extend(weather_data[index])  # 再把天气数据追加到check_in数据尾部
        
    np.savetxt('../datasets/Foursquare_new_check_in_data.txt', check_in_data,fmt='%f', delimiter= ' ')
    
