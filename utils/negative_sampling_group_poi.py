# sampling negative group-poi interection
import numpy as np
from tqdm import tqdm

read_ground_truth(file_path):
    groun_truth = set()
    truth_data = open(file_path, 'r').readlines()
    for eachline in truth_data:
        _,lid,_ = eachline.strip().split()
        lid = int(lid)
        groun_truth.add(lid)
    return groun_truth

def sample_negative_interection(dataset_name):
    if dataset_name == 'Fousquare':
        data_size = np.loadtxt('../datasets/Foursquare_FGRec/Foursquare_data_size.txt', dtype=np.int32)
        file = open('../datasets/group_poi/Fousquare_negative_group_poi.txt', mode='a')
    groun_truth = read_ground_truth('../datasets/:')
    for poi_id in tqdm(range(0, data_size[1])):

        pass
    else if dataset_name == 'Yelp':
        pass
    else if dataset_name == 'Gowalla':
        pass
