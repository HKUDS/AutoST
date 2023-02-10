# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 19:04:46 2022

@author: User
"""


import os
import torch
import random
import networkx as nx
import pandas as pd
import numpy as np
from torch.utils import data
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Batch
from itertools import repeat, product, chain
from collections import Counter, deque
from networkx.algorithms.traversal.breadth_first_search import generic_bfs_edges
import pickle

import torch_geometric.utils as tg_utils
import networkx as nx

def nx_to_graph_data_obj(g):
    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()
    # nodes
    nx_node_ids = [n_i for n_i in g.nodes()]  # contains list of nx node ids
    # print("nx_node_ids:", nx_node_ids)
    x = torch.tensor(np.ones(n_nodes).reshape(-1, 1), dtype=torch.float)
    # edges
    edges_list = []
    edge_features_list = []
    for node_1, node_2, attr_dict in g.edges(data=True):
        edge_feature = [attr_dict['weight'], attr_dict['date'], attr_dict['start'], attr_dict['end']]  # last 2 indicate self-loop
        # and masking
        edge_feature = np.array(edge_feature, dtype=int)
        # print("edge_feature:", edge_feature)
        # println()
        # convert nx node ids to data obj node index
        i = nx_node_ids.index(node_1)
        j = nx_node_ids.index(node_2)
        edges_list.append((i, j))
        edge_features_list.append(edge_feature)
        # edges_list.append((j, i))
        
        # edge_features_list.append(edge_feature)
        # print("edges_list:", edges_list)   
        
        # print("edge_features_list:", edge_features_list)
        # println()

    # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
    edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
    
    # print(edge_index)
    # print(edge_index.size())
    
    # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
    edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.float)
    # print(edge_attr.size())
    # println()
    # construct data obj
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return data


from loader import BioDataset_graphcl

class Dataset_graphcl(BioDataset_graphcl):
    # def __init__(self,root,data_type,empty=False,transform=None,pre_transform=None,pre_filter=None):
    def __init__(self,root,data_type,set_augMode, set_augStrength, empty=False,transform=None,pre_transform=None,pre_filter=None):
        # self.set_augMode('none')
        # self.set_augStrength(0.2)
        self.aug_mode = set_augMode
        self.aug_strength = set_augStrength
        # self.augmentations = [self.node_drop, self.subgraph, self.edge_pert, self.attr_mask, lambda x:x]
        self.set_generator(None, None)
        #super(Dataset_graphcl, self).__init__(root, data_type, empty, transform, pre_transform, pre_filter)
        super(BioDataset_graphcl, self).__init__(root, data_type, empty, transform, pre_transform, pre_filter)
        if not empty:
            self.data = nx_to_graph_data_obj(root)
            #self.data = root
            #self.data, self.slices = torch.load(self.processed_paths[0])
            #print("here")
            try:
                edge_index_neg = tg_utils.negative_sampling(self.data.edge_index, num_nodes=self.data.x.shape[0])
            except:
                edge_index_neg = tg_utils.negative_sampling(self.data.edge_index[:, :-1], num_nodes=self.data.x.shape[0])  # torch_geometric negative sampling bug
            self.data.edge_index_neg = edge_index_neg

            #print("inner_data:", self.data)
    # def set_augMode(self, aug_mode):
    #     self.aug_mode = aug_mode

    # def set_augStrength(self, aug_strength):
    #     self.aug_strength = aug_strength

    # def node_drop(self, data):
    #     node_num, _ = data.x.size()
    #     _, edge_num = data.edge_index.size()
    #     drop_num = int(node_num  * self.aug_strength)

    #     idx_perm = np.random.permutation(node_num)
    #     idx_nondrop = idx_perm[drop_num:].tolist()
    #     idx_nondrop.sort()

    #     edge_index, edge_attr = tg_utils.subgraph(idx_nondrop, data.edge_index, edge_attr=data.edge_attr, relabel_nodes=True, num_nodes=node_num)

    #     data.x = data.x[idx_nondrop]
    #     data.edge_index = edge_index
    #     data.edge_attr = edge_attr
    #     data.__num_nodes__, _ = data.x.shape
    #     return data

    # def edge_pert(self, data):
    #     node_num, _ = data.x.size()
    #     _, edge_num = data.edge_index.size()
    #     pert_num = int(edge_num * self.aug_strength)

    #     # delete edges
    #     idx_drop = np.random.choice(edge_num, (edge_num - pert_num), replace=False)
    #     edge_index = data.edge_index[:, idx_drop]
    #     edge_attr = data.edge_attr[idx_drop]

    #     # add edges
    #     adj = torch.ones((node_num, node_num))
    #     adj[edge_index[0], edge_index[1]] = 0
    #     edge_index_nonexist = adj.nonzero(as_tuple=False).t()
    #     idx_add = np.random.choice(edge_index_nonexist.shape[1], pert_num, replace=False)
    #     edge_index_add = edge_index_nonexist[:, idx_add]
    #     # random 9-dim edge_attr, for details please refer to https://github.com/snap-stanford/pretrain-gnns/issues/30
    #     edge_attr_add = torch.tensor( np.random.randint(2, size=(edge_index_add.shape[1], 7)), dtype=torch.float32 )
    #     edge_attr_add = torch.cat((edge_attr_add, torch.zeros((edge_attr_add.shape[0], 2))), dim=1)
    #     edge_index = torch.cat((edge_index, edge_index_add), dim=1)
    #     edge_attr = torch.cat((edge_attr, edge_attr_add), dim=0)

    #     data.edge_index = edge_index
    #     data.edge_attr = edge_attr
    #     return data

    # def attr_mask(self, data):
    #     node_num, _ = data.x.size()
    #     mask_num = int(node_num * self.aug_strength)
    #     _x = data.x.clone()

    #     token = data.x.mean(dim=0)
    #     idx_mask = np.random.choice(node_num, mask_num, replace=False)

    #     _x[idx_mask] = token
    #     data.x = _x
    #     return data

    # def subgraph(self, data):
    #     G = tg_utils.to_networkx(data)

    #     node_num, _ = data.x.size()
    #     _, edge_num = data.edge_index.size()
    #     sub_num = int(node_num * (1-self.aug_strength))

    #     idx_sub = [np.random.randint(node_num, size=1)[0]]
    #     idx_neigh = set([n for n in G.neighbors(idx_sub[-1])])

    #     while len(idx_sub) <= sub_num:
    #         if len(idx_neigh) == 0:
    #             idx_unsub = list(set([n for n in range(node_num)]).difference(set(idx_sub)))
    #             idx_neigh = set([np.random.choice(idx_unsub)])
    #         sample_node = np.random.choice(list(idx_neigh))

    #         idx_sub.append(sample_node)
    #         idx_neigh = idx_neigh.union(set([n for n in G.neighbors(idx_sub[-1])])).difference(set(idx_sub))

    #     idx_nondrop = idx_sub
    #     idx_nondrop.sort()

    #     edge_index, edge_attr = tg_utils.subgraph(idx_nondrop, data.edge_index, edge_attr=data.edge_attr, relabel_nodes=True, num_nodes=node_num)

    #     data.x = data.x[idx_nondrop]
    #     data.edge_index = edge_index
    #     data.edge_attr = edge_attr
    #     data.__num_nodes__, _ = data.x.shape
    #     return data

    def set_generator(self, generator1, generator2):
        self.generators = [generator1, generator2]

    def generator_generate(self, data, n_generator=1):
        with torch.no_grad():
            prob, edge_attr_pred = self.generators[n_generator-1].generate(data)
            prob = prob.numpy()

        node_num, _ = data.x.size()
        idx_sub = np.random.choice(node_num, 4, replace=False)
        idx_sub = [idx_sub[0], idx_sub[1], idx_sub[2], idx_sub[3]]
        idx_neigh = idx_sub
        idx_edge = []
        # weighted random walk
        for _ in range(10): # limit the walk within 10 steps
            _idx_neigh = []
            # online sampling based on p(v|v_c)
            for n in idx_neigh:
                try:
                    idx_neigh_n = np.random.choice(node_num, 1, p=prob[n])
                except:
                    idx_neigh_n = np.random.choice(node_num, 1)
                _idx_neigh += [idx_neigh_n[0]]
                idx_edge += [(n, idx_neigh_n[0]), (idx_neigh_n[0], n)]
            # get the new neighbors
            idx_neigh = _idx_neigh
            idx_sub += idx_neigh
        idx_sub = list(set(idx_sub))

        idx_sub.sort()
        idx_edge = list(set(idx_edge))
        idx_edge = torch.tensor(idx_edge).t()
        edge_attr = edge_attr_pred[idx_edge[0], idx_edge[1]]
        edge_index, edge_attr = tg_utils.subgraph(idx_sub, idx_edge, edge_attr=edge_attr, relabel_nodes=True, num_nodes=node_num)
        # edge_index, edge_attr = tg_utils.subgraph(idx_sub, idx_edge, edge_attr=edge_attr, relabel_nodes=False, num_nodes=node_num) # for visualization
        # data.idx_sub = idx_sub  # for visualization

        data.x = data.x[idx_sub]
        data.edge_index = edge_index
        data.edge_attr = edge_attr
        data.__num_nodes__, _ = data.x.shape
        print("gen_gen")
        return data

    def get(self, idx):
        print(0)
        data, data1, data2 = Data(), Data(), Data()
        # for key in self.data.keys:
        #     item = self.data[key]
        #     print(item.size())
        #     # item, slices = self.data[key], self.slices[key]
        #     # s = list(repeat(slice(None), item.dim()))
        #     # s[data.__cat_dim__(key, item)] = slice(slices[idx],
        #     #                                         slices[idx + 1])
        #     data[key], data1[key], data2[key] = item[s], item[s], item[s]
        # data, data1, data2 = custom_collate(self.data)

        if self.aug_mode == 'none':
            n_aug1, n_aug2 = 4, 4
            data1 = self.augmentations[n_aug1](data1)
            data2 = self.augmentations[n_aug2](data2)
        elif self.aug_mode == 'uniform':
            n_aug = np.random.choice(25, 1)[0]
            n_aug1, n_aug2 = n_aug//5, n_aug%5
            data1 = self.augmentations[n_aug1](data1)
            data2 = self.augmentations[n_aug2](data2)

        # generative model
        elif self.aug_mode == 'generative':
            data1 = self.generator_generate(data1, 1)
            data2 = self.generator_generate(data2, 2)

            try:
                edge_index_neg = tg_utils.negative_sampling(data.edge_index, num_nodes=data.x.shape[0])
            except:
                edge_index_neg = tg_utils.negative_sampling(data.edge_index[:,:-1], num_nodes=data.x.shape[0]) # torch_geometric negative sampling bug
            data.edge_index_neg = edge_index_neg
        print("get_get")
        return data, data1, data2


def custom_collate(data_list):
    batch = Batch([d[0] for d in data_list], follow_batch=['edge_index', 'edge_index_neg'])
    batch_1 = Batch.from_data_list([d[1] for d in data_list])
    batch_2 = Batch.from_data_list([d[2] for d in data_list])
    return batch, batch_1, batch_2


# class DataLoader(torch.utils.data.DataLoader):
#     def __init__(self, dataset, batch_size=1, shuffle=True,
#                  **kwargs):
#         super(DataLoader, self).__init__(dataset, batch_size, shuffle, collate_fn=custom_collate, **kwargs)
