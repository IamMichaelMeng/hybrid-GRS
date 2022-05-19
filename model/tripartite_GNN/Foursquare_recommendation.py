import sys
sys.path.append('../../')
import pdb
import utils.train_print as util
from model.tripartite_GNN.GAT import main as GAT_main

if __name__=='__main__':
    dataset= ['Foursquare']
    data_dir = "../../datasets/Foursquare/"
    size_file = data_dir + "Foursquare_data_size.txt"
    test_file = data_dir + "Foursquare_test.txt"

    num_users, num_pois, num_categories = open(size_file, 'r').readlines()[0].strip('\n').split()
    num_users, num_pois, num_categories = int(num_users), int(num_pois), int(num_categories)

    util.start_training('GAT')
    GAT = GAT_main()    
    util.end_training('GAT')

