import pdb
import torch
import numpy as np
from tqdm import tqdm
from utils.train_print import start_training
from utils.train_print import end_training
from utils.calculate_score import calculate_final_score_without_context
from model.Gowalla.GAT import main as GAT_main
from model.Gowalla.SBGNN import main as SBGNN_main
from model.Gowalla import SRGNN as SRGNN_main
from model.Gowalla.NGCF import main as NGCF_main
from utils.read_test_data import read_poi_context_intecrection,read_context_features,read_ground_truth
import datetime
import time
curr_time = datetime.datetime.now()

if __name__=='__main__':
    dataset= ['Gowalla']
    data_dir = "./datasets/Gowalla/"
    size_file = data_dir + "Gowalla_data_size.txt"
    test_file = data_dir + "Gowalla_test.txt"

    num_users, num_pois = open(size_file, 'r').readlines()[0].strip('\n').split()
    num_users, num_pois = int(num_users), int(num_pois)

    start_training('GAT')
    GAT = GAT_main()
    end_training('GAT')

    start_training('SBGNN')
    SBGNN, embedding_a, embedding_b = SBGNN_main()
    end_training('SBGNN')

    start_training('SRGNN')
    SRGNN = SRGNN_main.main()
    end_training('SRGNN')

    start_training('NGCF')
    #NGCF_CP = NGCF_main()
    end_training('NGCF_context_poi')


    print('traning ended...')
    print('prediction started...')
    
    # 读取测试数据
    group_emb = np.loadtxt('./datasets/group/Gowalla_group_emb.txt', dtype=np.float32)

    #context_features = read_context_features('./datasets/Gowalla/Gowalla_context_features.txt')

    test_data = read_ground_truth(test_file)
    all_user_ids = list(range(num_users))
    all_poi_ids = list(range(num_pois))

    # 构建测试数据集
    timestamp=datetime.datetime.strftime(curr_time,'%Y-%m-%d-%H-%M-%S')

    file = open('results/Gowalla_result_'+timestamp+'.txt', mode = 'a')
    for poi_id in tqdm(range(0, len(all_poi_ids))):
        SBGNN_poi_emb = embedding_b[poi_id]
        SBGNN_poi_emb = SBGNN_poi_emb.reshape(1, SBGNN_poi_emb.shape[0])
        SRGNN_poi_emb = SRGNN_main.get_poi_emb(SRGNN, poi_id)
        
        '''
        context_ids = list(interection_data[poi_id])
        context_fea = np.sum(context_features[context_ids], axis=0)
        weights = np.sum(NGCF_CP[context_ids].detach().numpy(), axis=0)
        context_feature = weights*context_fea
        '''
        overall_score = calculate_final_score_without_context(group_emb, SBGNN_poi_emb, SRGNN_poi_emb)
        temp = [str(poi_id)+' ', str(overall_score)+' ', '\n']
        file.writelines(temp)

    file.close()    

