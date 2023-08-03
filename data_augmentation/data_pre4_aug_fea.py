
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import warnings
warnings.filterwarnings('ignore')
import pickle
import dill
import networkx as nx
import torch
from torch_geometric.data import Data
import numpy as np
from torch_geometric.data import Batch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import InMemoryDataset
from model import GNN
from sklearn.metrics import roc_auc_score
import pandas as pd
from copy import deepcopy
from torch_geometric.nn import global_mean_pool, global_add_pool
from loader_aug import Dataset_graphcl
from loader import BioDataset_graphcl, BioDataset_graphcl1
from sklearn.cluster import KMeans 
from sklearn.metrics import adjusted_mutual_info_score
import json
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def load_data(file):
        data_load_file = []
        file_1 = open(file, "rb")
        data_load_file = pickle.load(file_1)
        return data_load_file

class graphcl(nn.Module):
    def __init__(self, gnn, emb_dim):
        super(graphcl, self).__init__()
        self.gnn = gnn
        self.pool = global_mean_pool
        self.projection_head = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU(inplace=True), nn.Linear(emb_dim, emb_dim))
        self.projection = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU(inplace=True), nn.Linear(emb_dim, emb_dim))

    def forward_cl(self, x, edge_index, edge_attr, edge_type, batch):
        
        x = self.gnn(x, edge_index.long(),edge_attr.long(), edge_type.long())
        x = self.pool(x, batch)
        x = self.projection_head(x)
        return x
    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / 0.5)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        return -torch.log(
            between_sim.sum(1)
            / (refl_sim.sum(1) + between_sim.sum(1)+ 1e-6))

    def loss_cl_1(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        simi = torch.exp(nn.CosineSimilarity()(z1,z2)/0.5)
        l1 = self.semi_loss(z1, z2)
        l2 = self.semi_loss(z2, z1)
        ret = (l1 + l2) * 0.5
        return ret
    



#reference: https://github.com/tkipf/gae; https://github.com/DaehanKim/vgae_pytorch
class vgae(nn.Module):
    def __init__(self, gnn, emb_dim):
        super(vgae, self).__init__()
        self.encoder = gnn
        self.encoder_mean = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU(inplace=True), nn.Linear(emb_dim, emb_dim))
        # make sure std is positive
        self.encoder_std = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU(inplace=True), nn.Linear(emb_dim, emb_dim), nn.Softplus())
        # only reconstruct first 7-dim, please refer to https://github.com/snap-stanford/pretrain-gnns/issues/30
        self.decoder = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(emb_dim, emb_dim), nn.ReLU(inplace=True), nn.Linear(emb_dim, 4), nn.Sigmoid())
        self.decoder_type = nn.Sequential(nn.ReLU(), nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, 1), nn.Sigmoid())
        self.decoder_edge = nn.Sequential(nn.ReLU(), nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, 1))
        self.decoder_type_1 = nn.Sequential(nn.ReLU(), nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, 1))

        self.bceloss = nn.BCELoss(reduction='none')
        self.pool = global_mean_pool
        self.add_pool = global_add_pool
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)

        # reconstruct 4-class & 3-class edge_attr for 1st & 2nd dimension
        self.decoder_1 = nn.Sequential(nn.ReLU(), nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, 4))
        self.decoder_2 = nn.Sequential(nn.ReLU(), nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, 4))
        self.crossentropyloss = nn.CrossEntropyLoss(reduction='none')

    def forward_encoder(self, x, edge_index, edge_attr, edge_type):
        
        x= self.encoder(x, edge_index,edge_attr, edge_type)
        
        
        x_mean = self.encoder_mean(x)
        x_std = self.encoder_std(x)
        gaussian_noise = torch.randn(x_mean.shape).to(x.device)
        x = gaussian_noise * x_std + x_mean
        return x, x_mean, x_std

    def forward_decoder(self, x, edge_index, edge_index_neg):
        
        eleWise_mul = x[edge_index[0]] * x[edge_index[1]]
        # print("eleWise_mul:",eleWise_mul.size())
        # print("decoder eleWise_mul:", eleWise_mul)
        edge_attr_pred = self.decoder(eleWise_mul)
        # print("decoder edge_attr_pred:", edge_attr_pred)
        # print("this is vgae part training")
        edge_pos = self.sigmoid( self.decoder_edge(eleWise_mul) ).squeeze()
        edge_neg = self.sigmoid( self.decoder_edge(x[edge_index_neg[0]] * x[edge_index_neg[1]]) ).squeeze()
        return edge_attr_pred, edge_pos, edge_neg
    def forward_decoder_type(self, x, edge_index):
        
        eleWise_mul = x[edge_index[0]] * x[edge_index[1]]
        # print("eleWise_mul:",eleWise_mul.size())
        edge_type_pred =  self.decoder_type(eleWise_mul)
        # print("this is vgae part training")
       
        return edge_type_pred  

    def loss_vgae(self, edge_attr_pred, edge_attr, edge_pos_pred, edge_neg_pred, edge_type_pred,edge_type_batch,edge_index_batch, edge_index_neg_batch, x_mean, x_std, batch, reward=None):
        # evaluate p(A|Z)
        num_edge, _ = edge_attr_pred.shape
        loss_rec = self.bceloss(edge_attr_pred.reshape(-1), edge_attr[:, :4].reshape(-1))
        loss_rec_type = self.bceloss(edge_type_pred.reshape(-1), edge_type_batch.reshape(-1))
        loss_rec = loss_rec.reshape((num_edge, -1)).sum(dim=1)
        loss_rec_type = loss_rec_type.reshape((num_edge, -1)).sum(dim=1)
        loss_edge_pos = self.bceloss(edge_pos_pred, torch.ones(edge_pos_pred.shape).to(edge_pos_pred.device))
        loss_edge_neg = self.bceloss(edge_neg_pred, torch.zeros(edge_neg_pred.shape).to(edge_neg_pred.device))
        loss_pos = loss_rec + loss_edge_pos
        loss_pos = self.pool(loss_pos, edge_index_batch)
        loss_neg = self.pool(loss_edge_neg, edge_index_neg_batch)
        loss_rec = loss_pos + loss_neg + loss_rec_type
        #print('loss_pos + loss_neg', loss_pos, loss_neg)
        if not reward is None:
            loss_rec = loss_rec * reward

        # evaluate p(Z|X,A)
        kl_divergence = - 0.5 * (1 + 2 * torch.log(x_std+ 1e-6) - x_mean**2 - x_std**2).sum(dim=1)
        # print("kl_divergence_1:",kl_divergence)
        kl_ones = torch.ones(kl_divergence.shape).to(kl_divergence.device)
        kl_divergence = self.pool(kl_divergence, batch)
        kl_double_norm = 1 / self.add_pool(kl_ones, batch)
        #print("kl_divergence_2:",kl_divergence)
        #print("kl_double_norm:",kl_double_norm)
        kl_divergence = kl_divergence * kl_double_norm
        # print("kl_divergence_3:",kl_divergence)
        # println()
        loss = (loss_rec + kl_divergence).mean()
        
        return loss, -(loss_edge_pos.mean()+loss_edge_neg.mean()).item()/2

    def generate(self, data):
        x, _, _ = self.forward_encoder(data.x, data.edge_index, data.edge_attr, data.edge_type)
        eleWise_mul = torch.einsum('nd,md->nmd', x, x)
        edge_type_prob_1 = self.decoder_type_1(eleWise_mul)
        edge_type_rand_1 = torch.rand((edge_type_prob_1.shape[0], edge_type_prob_1.shape[1]))
        edge_type_pred_1 = torch.zeros((edge_type_prob_1.shape[0], edge_type_prob_1.shape[1]), dtype=torch.int64)
        # print("edge_type_pred_1:", edge_type_pred_1.size())
        # println()
        for n in range(2):
            edge_type_pred_1[edge_type_rand_1 >= edge_type_prob_1[:, n]] = n + 1
            edge_type_rand_1 -= edge_type_prob_1[:, n]
        prob = self.decoder_edge(eleWise_mul).squeeze()
        prob = torch.exp(prob)
        prob[torch.isinf(prob)] = 1e10
        prob[list(range(x.shape[0])), list(range(x.shape[0]))] = 0
        prob = torch.einsum('nm,n->nm', prob, 1 / prob.sum(dim=1))

        # sparsify
        prob[prob < 1e-1] = 0
        prob[prob.sum(dim=1) == 0] = 1
        prob[list(range(x.shape[0])), list(range(x.shape[0]))] = 0
        prob = torch.einsum('nm,n->nm', prob, 1 / prob.sum(dim=1))

        # predict 4-class & 3-class edge_attr for 1st & 2nd dimension
        edge_attr_prob_1 = self.softmax(self.decoder_1(eleWise_mul))
        edge_attr_rand_1 = torch.rand((edge_attr_prob_1.shape[0], edge_attr_prob_1.shape[1]))
        edge_attr_pred_1 = torch.zeros((edge_attr_prob_1.shape[0], edge_attr_prob_1.shape[1]), dtype=torch.int64)
        for n in range(3):
            edge_attr_pred_1[edge_attr_rand_1 >= edge_attr_prob_1[:, :, n]] = n + 1
            edge_attr_rand_1 -= edge_attr_prob_1[:, :, n]

        edge_attr_prob_2 = self.softmax(self.decoder_2(eleWise_mul))
        # print("edge_attr_prob_2:",edge_attr_prob_2)
        # println()
        edge_attr_rand_2 = torch.rand((edge_attr_prob_2.shape[0], edge_attr_prob_2.shape[1]))
        edge_attr_pred_2 = torch.zeros((edge_attr_prob_2.shape[0], edge_attr_prob_2.shape[1]), dtype=torch.int64)
        for n in range(2):
            edge_attr_pred_2[edge_attr_rand_2 >= edge_attr_prob_2[:, :, n]] = n + 1
            edge_attr_rand_2 -= edge_attr_prob_2[:, :, n]
        edge_attr_pred = torch.cat((edge_attr_pred_1.reshape((edge_attr_prob_1.shape[0], edge_attr_prob_1.shape[1], 1)),
                                    edge_attr_pred_2.reshape(
                                        (edge_attr_prob_2.shape[0], edge_attr_prob_2.shape[1], 1)),edge_attr_pred_2.reshape(
                                        (edge_attr_prob_2.shape[0], edge_attr_prob_2.shape[1], 1)),edge_attr_pred_2.reshape(
                                        (edge_attr_prob_2.shape[0], edge_attr_prob_2.shape[1], 1))), dim=2)
        edge_type_pred = edge_type_pred_1.view(edge_type_pred_1.size()[0],edge_type_pred_1.size()[1], 1 )
        return prob, edge_attr_pred, edge_type_pred



def custom_collate(data_list):
    batch = Batch.from_data_list([d[0] for d in data_list], follow_batch=['edge_index', 'edge_index_neg'])
    batch_1 = Batch.from_data_list([d[1] for d in data_list])
    batch_2 = Batch.from_data_list([d[2] for d in data_list])
    return batch, batch_1, batch_2


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True,
                  **kwargs):
        
        super(DataLoader, self).__init__(dataset, batch_size, shuffle, collate_fn=custom_collate, **kwargs)


def train(args, loader, model_cl, optimizer_cl, model_1, optimizer_1, model_2, optimizer_2, model_ib, optimizer_ib, gamma, device, flag):
    pretrain_loss, generative_loss = 0, 0
    link_loss = 0
    # for step, batch in enumerate(tqdm(loader, desc="Iteration")):
    step0 = 0
    for step, batch in enumerate(loader):

        batch, batch1, batch2 = batch
        batch, batch1, batch2 = batch.to(device), batch1.to(device), batch2.to(device)
        
        # 1. graphcl
        optimizer_cl.zero_grad()
        
        x1 = model_cl.forward_cl(batch1.x, batch1.edge_index, batch1.edge_attr,batch1.edge_type, batch1.batch)
        x2 = model_cl.forward_cl(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.edge_type,batch2.batch)
        # x = model_cl.forward_cl(batch.x, batch.edge_index, batch.edge_attr, batch.batch)


        # loss_cl = model_cl.loss_cl(x1, x2, mean=False)
        loss_cl = model_cl.loss_cl_1(x1, x2, mean=False)
        similarity = torch.cosine_similarity(x1, x2, dim=1)
        
        loss = loss_cl.mean()

        loss.backward()
        
        optimizer_cl.step()
        pretrain_loss += float(loss.item())
        
        # information bottleneck
        optimizer_ib.zero_grad()
        _x1 = model_ib.forward_cl(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.edge_type, batch1.batch)
        _x2 = model_ib.forward_cl(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.edge_type,batch2.batch)
        loss_ib = model_ib.loss_cl_1(_x1, x1.detach(), mean=False) + model_ib.loss_cl_1(_x2, x2.detach(), mean=False)
        loss = loss_ib.mean()
        loss.backward()
        
        optimizer_ib.step()
        # print("loss_ib:", loss)
        # println()
        # reward for joao
        loss_cl = (1 - gamma) * loss_cl.detach() + gamma * loss_ib.detach()
        # loss_cl = (1 - gamma) * loss_cl.detach()
        # loss_cl = loss_cl.detach()
        # print("^^^^:", loss_cl.mean(), loss_cl, loss_cl.std())
        loss_cl = loss_cl -loss_cl.mean()

        loss_cl[loss_cl > 0] = 1
        loss_cl[loss_cl <= 0] = 0.01  # weaken the reward for low cl loss
        # print("loss_cl__2:", loss_cl)
        # println()
        # 2. joao
        optimizer_1.zero_grad()
        optimizer_2.zero_grad()
        
        
        x, x_mean, x_std = model_1.forward_encoder(batch.x, batch.edge_index,batch.edge_attr, batch.edge_type)
        # print("x_shape_model1:", x[0])
        # print("output x:", x)
        edge_attr_pred, edge_pos_pred, edge_neg_pred = model_1.forward_decoder(x, batch.edge_index, batch.edge_index_neg)
        edge_type_pred = model_1.forward_decoder_type(x, batch.edge_index)
        

        # pritnln()

        
        # *loss_con_1.item()
        #adding reward signal
        w = 0.7
        gam = 0.05
        re = w*(1-similarity.item())+(1-w)*loss_cl
        # re = loss_cl
        # print("re:",re)
        # re = loss_cl
        loss_1, link_loss_1 = model_1.loss_vgae(edge_attr_pred, batch.edge_attr, edge_pos_pred, edge_neg_pred, edge_type_pred,batch.edge_type, batch.edge_index_batch, batch.edge_index_neg_batch, x_mean, x_std, batch.batch, reward=re)
        # print("loss_1:", loss_1)
        # print("done model_1")
        x, x_mean, x_std = model_2.forward_encoder(batch.x, batch.edge_index, batch.edge_attr, batch.edge_type)
        # print("x_shape_model2:", x[0])
        edge_attr_pred, edge_pos_pred, edge_neg_pred = model_2.forward_decoder(x, batch.edge_index, batch.edge_index_neg)
        edge_type_pred = model_2.forward_decoder_type(x, batch.edge_index)
        


        #adding reward signal
        w = 0.7
        gam = 0.05
        re = w*(1-similarity.item())+(1-w)*loss_cl
        # re = loss_cl

        # w = 0.5
        # re = w*(1-similarity.item())+(1-w)*loss_cl
        # re = loss_cl
        loss_2, link_loss_2 = model_2.loss_vgae(edge_attr_pred, batch.edge_attr, edge_pos_pred, edge_neg_pred,edge_type_pred,batch.edge_type, batch.edge_index_batch, batch.edge_index_neg_batch, x_mean, x_std, batch.batch, reward=re)
        # print("loss_2:", loss_2)
        loss = loss_1 + loss_2
        
        loss.backward()
        
        optimizer_1.step()
        optimizer_2.step()
        generative_loss += float(loss.item())

        link_loss += (link_loss_1+link_loss_2)/2
        step0 = step

    return pretrain_loss/(step+1), generative_loss/(step+1), link_loss/(step+1)
def test(model, x, edge_index,edge_attr, edge_type):   
    model.eval()
    x, x_mean, x_std = model.forward_encoder(x, edge_index,edge_attr, edge_type)
    print("Node embedding dimension:", x.size())
    file=open(r"../data/web_vector_closs_w_0.7.pickle","wb")
    # file=open(r"../data/web_vector_closs.pickle","wb")
    # file=open(r"../data/web_vector_closs_infomin.pickle","wb"))
    pickle.dump(x,file) #storing_list
    file.close()
    
    # return label_classification(z, y, ratio=0.1),label_classification(z_1, y, ratio=0.1)
    print("--This is the end of whole process---")    

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0.01,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=96,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gcn")
    parser.add_argument('--num_workers', type=int, default = 0, help='number of workers for dataset loading')
    parser.add_argument('--aug_mode', type=str, default = 'generative') 
    parser.add_argument('--aug_strength', type=float, default = 0.2)
    parser.add_argument('--resume', type=int, default=0)
    parser.add_argument('--gamma', type=float, default=0.1)
    # parser.add_argument('--dataset', type=str, default='dataset_new_2')
    args = parser.parse_args()
    # device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device('cuda:'+ str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    root_unsupervised = '../data/dataset/'

    dataset = BioDataset_graphcl(root_unsupervised, data_type='unsupervised')
    # print("dataset:", dataset)
    data = dataset.data

    for subg_idx in range(30):
        sample_size = 800
        G = nx.Graph()
        G.add_edges_from(list(zip(data.edge_index.numpy()[0],data.edge_index.numpy()[1])))
        # print("G:", G)
        S = G.subgraph(np.random.permutation(G.number_of_nodes())[:sample_size])
        x = data.x[np.array(S.nodes())]
        edge_type = data.edge_type[np.array(S.edges())[:,0]]
        edge_attr = data.edge_attr[np.array(S.edges())[:,0]]
        S = nx.relabel.convert_node_labels_to_integers(S, first_label=0, ordering='default')
        edge_index = torch.tensor(np.array(S.edges()).T)
        
        sdata = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_type = edge_type)
        print("sdata:", sdata)
        from itertools import product
        def get_data(d):
            data_list = [0]
            data_list[0] = d
            data = data_list[0]
            keys = data_list[0].keys
            data = data_list[0].__class__()
            for key in keys:
                data[key] = []
            slices = {key: [0] for key in keys}
            for item, key in product(data_list, keys):
                data[key].append(item[key])
                if torch.is_tensor(item[key]):
                    s = slices[key][-1] + item[key].size(item.__cat_dim__(key, item[key]))
                else:
                    s = slices[key][-1] + 1
                slices[key].append(s)
            if hasattr(data_list[0], '__num_nodes__'):
                data.__num_nodes__ = []
                for item in data_list:
                    data.__num_nodes__.append(item.num_nodes)
            for key in keys:
                item = data_list[0][key]
                if torch.is_tensor(item):
                    data[key] = torch.cat(data[key],dim=data.__cat_dim__(key, item))
                elif isinstance(item, int) or isinstance(item, float):
                    data[key] = torch.tensor(data[key])
                slices[key] = torch.tensor(slices[key], dtype=torch.long)
            com = (data, slices)
            return com



        sdata = get_data(sdata)
        torch.save(sdata,'../data/dataset/processed/sdata.pt')
        root = "../data/dataset/"
        # sdataset = BioDataset_graphcl('../data/dataset/processed/sdata.pt', data_type='unsupervised')
        
        sdataset = BioDataset_graphcl1(root, data_type='unsupervised')
        sdataset.set_augMode(args.aug_mode)
        sdataset.set_augStrength(args.aug_strength)
        loader = DataLoader(sdataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
        # print(loader)
        
        # pritnln()
        # set up graphcl model
        gnn = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type)
        model = graphcl(gnn, args.emb_dim)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

        # information bottleneck
        gnn_ib = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type)
        model_ib = graphcl(gnn_ib, args.emb_dim)
        model_ib.to(device)
        optimizer_ib = optim.Adam(model_ib.parameters(), lr=args.lr, weight_decay=args.decay)

        # set up vgae model 1
        gnn_generative_1 = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio,
                               gnn_type=args.gnn_type)
        model_generative_1 = vgae(gnn_generative_1, args.emb_dim)
        model_generative_1.to(device)
        optimizer_generative_1 = optim.Adam(model_generative_1.parameters(), lr=args.lr, weight_decay=args.decay)

        # set up vgae model 2
        gnn_generative_2 = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio,
                               gnn_type=args.gnn_type)
        model_generative_2 = vgae(gnn_generative_2, args.emb_dim)
        model_generative_2.to(device)
        optimizer_generative_2 = optim.Adam(model_generative_2.parameters(), lr=args.lr, weight_decay=args.decay)

        # start training
        model.train(), model_generative_1.train(), model_generative_2.train()
        if args.resume == 0:
            torch.save({'graphcl': model.state_dict(), 'graphcl_opt': optimizer.state_dict(),
                        'generator_1': model_generative_1.state_dict(),
                        'generator_1_opt': optimizer_generative_1.state_dict(),
                        'generator_2': model_generative_2.state_dict(),
                        'generator_2_opt': optimizer_generative_2.state_dict()},
                        './weights_generative/checkpoint_ibgamma' + str(args.gamma) + '_0.pth')

        gamma = args.gamma
        flag = True
        for epoch in range(args.resume + 1, args.epochs + 1):
            #if epoch == args.epochs:
             #   flag = True
            loader.dataset.set_generator(deepcopy(model_generative_1).cpu(), deepcopy(model_generative_2).cpu())
            pretrain_loss, generative_loss, link_loss = train(args, loader, model, optimizer, model_generative_1,
                                                              optimizer_generative_1, model_generative_2,
                                                              optimizer_generative_2, model_ib, optimizer_ib, gamma, device, flag)
        
            if epoch % 1 == 0:
                print('\n', subg_idx, epoch, generative_loss, link_loss)

    
            # torch.save({'graphcl':model.state_dict(), 'graphcl_opt': optimizer.state_dict(), 'graphcl_ib':model_ib.state_dict(), 'graphcl_ib_opt': optimizer_ib.state_dict(), 'generator_1':model_generative_1.state_dict(), 'generator_1_opt':optimizer_generative_1.state_dict(), 'generator_2':model_generative_2.state_dict(), 'generator_2_opt':optimizer_generative_2.state_dict()}, './weights_generative/checkpoint_ibgamma'+str(args.gamma)+'_'+str(epoch)+'.pth')
    print("-------Start Testing--------")
    sample_size = 1388
    G = nx.Graph()
    G.add_edges_from(list(zip(data.edge_index.numpy()[0],data.edge_index.numpy()[1])))
    # print("G:", G)
    S = G.subgraph(np.random.permutation(G.number_of_nodes())[:sample_size])
    x = data.x[np.array(S.nodes())]
    edge_type = data.edge_type[np.array(S.edges())[:,0]]
    edge_attr = data.edge_attr[np.array(S.edges())[:,0]]
    S = nx.relabel.convert_node_labels_to_integers(S, first_label=0, ordering='default')
    edge_index = torch.tensor(np.array(S.edges()).T)
    test(model_generative_1, x.to(device), edge_index.to(device),edge_attr.to(device), edge_type.to(device))
import time
if __name__ == '__main__': 
    start_time  = time.time()       
    main()

    end_time = time.time()
    print((end_time-start_time)/100)
    # data augmentation


















