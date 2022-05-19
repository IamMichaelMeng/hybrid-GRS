import pdb
import torch
import numpy as np
from tqdm import tqdm
from utils.train_print import start_training
from utils.train_print import end_training
from utils.read_test_data import read_poi_context_intecrection,read_context_features,read_ground_truth
from model.Foursquare.GAT import main as GAT_main
from model.Foursquare.SBGNN import main as SBGNN_main
from model.Foursquare import SRGNN as SRGNN_main
from model.Foursquare.NGCF import main as NGCF_main
from sklearn.preprocessing import normalize

def calculate_final_score(group_emb, SBGNN_poi_emb, SRGNN_poi_emb, context_feature):
    randindex = torch.randperm(SBGNN_poi_emb.shape[1])

    SRGNN_poi_emb = SRGNN_poi_emb[0][randindex].reshape(1, SBGNN_poi_emb.shape[1])
    group_emb = group_emb[randindex].reshape(1, SBGNN_poi_emb.shape[1])
    group_emb = normalize(group_emb, axis=0, norm='max') # 归一化处理
    poi_emb = SBGNN_poi_emb + SRGNN_poi_emb
    
    # 计算group-poi预测分数
    score = float(torch.sigmoid(torch.mm(torch.from_numpy(group_emb), poi_emb.T))) 
    return score

if __name__=='__main__':
    dataset= ['Foursquare']
    data_dir = "./datasets/Foursquare/"
    size_file = data_dir + "Foursquare_data_size.txt"
    test_file = data_dir + "Foursquare_test.txt"
    
    num_users, num_pois, num_categories = open(size_file, 'r').readlines()[0].strip('\n').split()
    num_users, num_pois, num_categories = int(num_users), int(num_pois), int(num_categories)
    
    start_training('GAT')
    #GAT = GAT_main()    
    end_training('GAT')

    start_training('SBGNN')
    #SBGNN, embedding_a, embedding_b = SBGNN_main()
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
    group_emb = np.loadtxt('./datasets/group_emb.txt', dtype=np.float32)
    interection_data = read_poi_context_intecrection('./datasets/Foursquare/Foursquare_check_ins_tripartite_graph_new.txt')
    context_features = read_context_features('./datasets/Foursquare/Foursquare_context_features.txt')

    test_data = read_ground_truth(test_file)
    all_user_ids = list(range(num_users))
    all_poi_ids = list(range(num_pois))

    # 构建测试数据集
    file = open('results/Foursquare_result_without_context.txt', mode = 'a') 
    for poi_id in tqdm(range(0, len(all_poi_ids))):
        SBGNN_poi_emb = embedding_b[poi_id]
        SBGNN_poi_emb = SBGNN_poi_emb.reshape(1, SBGNN_poi_emb.shape[0])
        SRGNN_poi_emb = SRGNN_main.get_poi_emb(SRGNN, poi_id)

        context_ids = list(interection_data[poi_id])
        context_fea = np.sum(context_features[context_ids], axis=0)
        weights = np.sum(NGCF_CP[context_ids].detach().numpy(), axis=0)
        context_feature = weights*context_fea
        overall_score = calculate_final_score(group_emb, SBGNN_poi_emb, SRGNN_poi_emb, context_feature)
        temp = [str(poi_id)+' ', str(overall_score)+' ', '\n']
        file.writelines(temp)

    file.close()
