import sys
sys.path.append('../../')
import pdb
import torch
from tqdm import tqdm
import torch_scatter
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch.nn import Parameter, Linear
import torch_geometric.utils as pyg_utils
from typing import Union, Tuple, Optional
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType, OptTensor)
from utils.create_node_emb import create_user_emb
from utils.create_social_network import create_tripartite_graph
from utils.metrics import embedding_learning_accuracy
from utils.calculate_node_influence import get_node_influence 
from utils.aggregate_user_emb import aggregate_user_embedding
from utils.objectview import Objectview
from utils.takeOrder import takeOrder
from utils.pos_and_nega_edges import *
from utils.optimizer import build_optimizer



class GNNStack(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args, emb=False):
        super(GNNStack, self).__init__()
        conv_model = self.build_conv_model(args.model_type)
        self.convs = nn.ModuleList()
        self.convs.append(conv_model(input_dim, hidden_dim, heads=args.heads))
        assert (args.num_layers >= 1), 'Number of layers is not >=1'
        for l in range(args.num_layers-1):
            self.convs.append(conv_model(args.heads * hidden_dim, hidden_dim,
                                         heads=args.heads))

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(args.heads * hidden_dim, hidden_dim), nn.Dropout(args.dropout),
            nn.Linear(hidden_dim, output_dim))

        self.dropout = args.dropout
        self.num_layers = args.num_layers

        self.emb = emb

    def build_conv_model(self, model_type):
        if model_type == 'GAT':
            return GAT

    def forward(self, emb, train_edge):
        x = emb.weight.data
        edge_index = train_edge
        
        # forward  processing
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout)
        x = self.post_mp(x)

        # get embedding after forward
        emb_set_u = x[train_edge[0]]
        emb_set_v = x[train_edge[1]]

        # dot product the embeddings between each node pair
        #dot_prod = torch.sum(emb_set_u + emb_set_v, dim=-1)
        dot_prod = torch.sum(emb_set_u * emb_set_v, dim=-1)
        
        # Feed the dot product result into sigmoid
        sig = torch.sigmoid(dot_prod)
        
        return sig, x 

    def loss(self, pred, label):
        return F.nll_loss(pred, label)



class GAT(MessagePassing):

    def __init__(self, in_channels, out_channels, heads = 10,
                 negative_slope = 0.2, dropout = 0., **kwargs):
        super(GAT, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin_l = None
        self.lin_r = None
        self.att_l = None
        self.att_r = None


        self.lin_l = nn.Linear(self.in_channels, self.heads*self.out_channels)
        self.lin_r = self.lin_l
        self.att_l = nn.Parameter(torch.randn(heads, self.out_channels)) #Head x C
        self.att_r = nn.Parameter(torch.randn(heads, self.out_channels)) #Head x C

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)

    def forward(self, x, edge_index, size = None):
        
        H, C = self.heads, self.out_channels
        
        x_l = self.lin_l(x).view(-1, H, C) #N x H x C
        x_r = self.lin_r(x).view(-1, H, C) #N x H x C
                                                
        alpha_l = self.att_l.unsqueeze(0)*x_l #1 x H x C * N x H x C 
        alpha_r = self.att_r.unsqueeze(0)*x_r #1 x H x C * N x H x C
        
        out = self.propagate(edge_index, x=(x_l, x_r), alpha=(alpha_l, alpha_r))
        out = out.view(-1, H*C)

        return out


    def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i):
        alpha_ij = F.leaky_relu(alpha_i + alpha_j, negative_slope=self.negative_slope)
        
        if ptr is not None:
            alpha_ij = softmax(alpha_ij, ptr)
        else:
            alpha_ij = softmax(alpha_ij, index)
            
        alpha_ij = F.dropout(alpha_ij, p=self.dropout)
        out = x_j * alpha_ij
        return out

    def aggregate(self, inputs, index, dim_size = None):

        node_dim = self.node_dim
        out = torch_scatter.scatter(inputs, index, dim=node_dim, reduce="sum")
        return out

def train(graph, emb, args, train_edge, train_label, loss_fn):
    # define model
    model = GNNStack(args.num_node_features, args.hidden_dim, args.output_dim, args)
    scheduler, optimizer = build_optimizer(args, model.parameters())

    # train 
    losses = []
    best_loss = 10000
    best_accuracy = 0.0
    best_epoch = 0
    best_embedding = []
    print('GAT training started...')
    for epoch in tqdm(range(args.epochs)):
        model.train()
        optimizer.zero_grad()
        pred, embeddings = model(emb, train_edge) 
        loss = loss_fn(pred, train_label)
        accuracy = embedding_learning_accuracy(pred, train_label) 
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        
        if loss < best_loss:
            best_loss = loss
            best_epoch = epoch
            best_accuracy = accuracy
            best_embedding = embeddings
        losses.append(loss)        
    print(f'best result is: \nEpoch:{best_epoch}\nLoss:{best_loss}\nAccuracy:{best_accuracy}{best_loss}')

    return best_embedding


def main():
    # 将user-context和context-poi两个数据集读取出来合并到一起
    user_context = np.loadtxt('../../datasets/Foursquare/Foursquare_user_context.txt', dtype=np.int32).tolist()
    context_poi = np.loadtxt('../../datasets/Foursquare/Foursquare_context_poi.txt', dtype=np.int32).tolist() 
    user_context_poi = []
    user_context_poi.extend(user_context)
    user_context_poi.extend(context_poi)

    # 读取连接数据，构建social_network图
    graph = create_tripartite_graph(user_context_poi)
    pdb.set_trace()

    # 构建node embedding
    user_info = np.loadtxt('../../datasets/Foursquare/Foursquare_user_features.txt', dtype=np.int32)
    num_node = len(user_info)
    embedding_dim = len(user_info[0][0:-1])
    emb = create_user_emb(num_node, embedding_dim)

    # create positive and negative edges
    pos_edge_list = graph_to_edge_list(graph)
    pos_edge_index = edge_list_to_tensor(pos_edge_list)

    neg_edge_list = sample_negative_edges(graph, len(pos_edge_list))
    neg_edge_index = edge_list_to_tensor(neg_edge_list)

    # assign loss_fn and sigmoid
    loss_fn = nn.BCELoss()
    sigmoid = nn.Sigmoid()

    # Generate the positive and negative labels
    pos_label = torch.ones(pos_edge_index.shape[1], )
    neg_label = torch.zeros(neg_edge_index.shape[1], )

    # Concat positive and negative labels into one tensor
    train_label = torch.cat([pos_label, neg_label], dim=0)

    # Concat positive and negative edges into one tensor
    train_edge = torch.cat([pos_edge_index, neg_edge_index], dim=1)

    # mdoel settings
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = {
    'model_type': 'GAT',
    'device': device,
    'num_layers': 2,
    'heads': 1,
    'batch_size':32,
    'hidden_dim': 32,
    'dropout': 0.5,
    'lr': 0.01,
    'epochs': 100,
    'num_node_features': embedding_dim,
    'output_dim':10,
    'weight_decay':5e-4,
    'opt':'adam',
    'opt_scheduler':'none',
    }
    args = Objectview(args)
    best_embedding = train(graph, emb, args, train_edge, train_label, loss_fn)
    np.savetxt('datasets/GAT_trained_user_features.txt', best_embedding.detach().numpy())

    # aggregate user emb to get group emb
    group_emb = aggregate_user_embedding(node_influence)
    np.savetxt('datasets/group_emb.txt',  group_emb, fmt='%f')
    pass
