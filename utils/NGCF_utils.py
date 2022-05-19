# NGCF模型专用工具函数
import pdb
from collections import defaultdict
import pandas as pd
from torch.utils.data import Dataset

class Foursquare(Dataset):

    def __init__(self,rt):
        super(Dataset,self).__init__()
        self.uId = list(rt['userId'])
        self.iId = list(rt['itemId'])
        self.rt = list(rt['frequency'])

    def __len__(self):
        return len(self.uId)

    def __getitem__(self, item):
        return (self.uId[item],self.iId[item],self.rt[item])

def read_checkin_data(path):
    df = pd.read_table(path,sep='\t',names=['userId','itemId','frequency'])
    return df

def read_checkin_data_include_context(path):
    df = pd.read_table(path, sep=' ', names=['userId','itemId','frequency','contextId'])
    cluster = [df.itemId, df.contextId, df.frequency]
    df = pd.concat(cluster, axis=1)
    df = df.rename(columns={'itemId':'userId'})
    df = df.rename(columns={'contextId':'itemId'})
    return df
 
