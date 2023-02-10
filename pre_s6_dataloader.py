
import torch
from torch_geometric.data import Data
from itertools import product
import numpy as np
import pandas as pd
from torch import nn
import pickle

def load_data(file):
        data_load_file = []
        file_1 = open(file, "rb")
        data_load_file = pickle.load(file_1)
        return data_load_file
resolution = 500
linear = nn.Linear(512, 96)
# node_features_1 = load_data("../data/region_spatial_refine.pickle") 
# node_features = load_data("../data/region_spatial_refine.pickle")
node_features = load_data("../data/region_spatial_refine.pickle")
region_poi_vec = load_data("../data/reg_poi_vec.pickle")
region_trans = linear(region_poi_vec)
# print(region_trans)
# print(region_trans.size())
# pritnln()
# reg_com_poi_cat_spatial.pickle
# node_features = load_data("../data/reg_vector_dict.pickle") 

# node_features_2 = load_data("../data/reg_flow_dict_vec_4.pickle")
# print(type(node_features_1))
# print(type(node_features_2))
# node_feature = [torch.mean(torch.cat((i[1],j),axis= 1)) for i,j in zip(node_features_1.items(),node_features_2)]

# print(node_feature[0].size())
# print(len(node_feature))
# println()
def nx_to_graph_data_obj(g):
    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()
    # nodes
    nx_node_ids = [n_i for n_i in g.nodes()]  # contains list of nx node ids
    # print("nx_node_ids:", nx_node_ids)
    # n = np.array([nx_node_ids.index(n_i) for n_i in g.nodes()])
    x_ = torch.tensor(np.ones(n_nodes).reshape(-1, 1), dtype=torch.float)
    # print("nx_node_ids:",nx_node_ids)
    n_nodes = [int(item.split("_")[1]) for item in nx_node_ids]
    # print("n_nodes:",n_nodes)
    # print(len(n_nodes))
    # x = torch.tensor([torch.squeeze(node_features[item],0).tolist() for item in n_nodes])
    # x = torch.tensor([torch.squeeze(node_features[item],0).tolist() for item in n_nodes])
    x = torch.tensor([region_trans[item].tolist() for item in n_nodes])
    # print("x:",x.size())
    # print(x_.size())
    # printnln()
    file=open(r"../data/nodes_new_{}.pickle".format(7),"wb")
    pickle.dump(nx_node_ids,file) #storing_list
    file.close()
    # x = torch.tensor(n.reshape(-1, 1), dtype=torch.float)
    # print("x:", x)
    # println()
    # edges
    edges_list = []
    edge_features_list = []
    for node_1, node_2, attr_dict in g.edges(data=True):
        # print("attr_dict:", attr_dict)
        # print("node_1:", node_1)
        # print("node_2:", node_2)
        edge_feature = [attr_dict['weight'], attr_dict['date'], nx_node_ids.index(attr_dict['start']), nx_node_ids.index(attr_dict['end'])]  # last 2 indicate self-loop
        # and masking
        edge_feature = np.array(edge_feature, dtype=int)
        # convert nx node ids to data obj node index
        i = nx_node_ids.index(node_1)
        j = nx_node_ids.index(node_2)
        edges_list.append((i, j))
        edge_features_list.append(edge_feature)
    # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
    edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
    # print("edge_index:", edge_index)
    
    # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
    edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.float)
    # print("edge_attr:", edge_attr.size())
    # println()
    # construct data obj
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

# print(data)
# println()


# edge_index = torch.tensor([
#     [3, 1, 1, 2],
#     [1, 3, 2, 1]], dtype=torch.long)
# x = torch.tensor([[-1],
#                   [0],
#                   [1]], dtype=torch.float)

# d = Data(x=x, edge_index=edge_index)
# print(type(d)) # # <class 'torch_geometric.data.data.Data'>
def get_data(d):
    data_list = [0]
    data_list[0] = d
    # print("data_list:", data_list)
    data = data_list[0]
    # print(data)
    # println()
    keys = data_list[0].keys
    # data->Data()
    data = data_list[0].__class__()
    # print("data:", data_list[0])
    # println()
    # print(data_list[0].keys) # ['x', 'edge_index']
    # print(type(data)) # <class 'torch_geometric.data.data.Data'>
    # print("before_data:", data_list)
    for key in keys:
        data[key] = []
    # print("initial_data:", data) # Data(edge_index=[0], x=[0])
    slices = {key: [0] for key in keys}
    # print(slices) # {'x': [0], 'edge_index': [0]}
    # print("slices:", slices)
    for item, key in product(data_list, keys):
        # print("111:", item, key)
        # print("222:", item[key])
        data[key].append(item[key])
        # print("middle_data:", data)
        # println()
        if torch.is_tensor(item[key]):
            # print("slices[key]:", slices[key][-1])
            # print("item[key]:", item.__cat_dim__(key, item[key]))
            # print("%%%:", item[key].size(item.__cat_dim__(key, item[key])))
            # 
            s = slices[key][-1] + item[key].size(item.__cat_dim__(key, item[key]))
            # print("s^^^:", s)
        else:
            s = slices[key][-1] + 1
            # print("s***:", s)
        slices[key].append(s)
        # print("slices_after:", slices)
    
    # print("final_data:", data)
    # println()
    
    
    if hasattr(data_list[0], '__num_nodes__'):
        data.__num_nodes__ = []
        for item in data_list:
            data.__num_nodes__.append(item.num_nodes)
    
    for key in keys:
        item = data_list[0][key]
        if torch.is_tensor(item):
            print("__data[key]:", len(data[key]))
            print("tmp:", data.__cat_dim__(key, item))
            
            data[key] = torch.cat(data[key],
                                  dim=data.__cat_dim__(key, item))
            print("data[key]__:", len(data[key]))
            
        elif isinstance(item, int) or isinstance(item, float):
            data[key] = torch.tensor(data[key])
    
        slices[key] = torch.tensor(slices[key], dtype=torch.long)
        
        
    # print("data:", data)
    # print("slices:", slices)   
    com = (data, slices)
    # print(com)
    return com
# import os.path as osp
# def get(idx):
#     data = torch.load(osp.join("../data/dataset/processed/", 'dataset_{}.pt',format(idx)))
#     return data
# hy_graph = load_data("../data/hy_new_s.pickle")
hy_graph = load_data("../data/hy_new_test_60.pickle")
d = nx_to_graph_data_obj(hy_graph)
com = get_data(d)
torch.save(com,'./data/dataset/processed/dataset_new_60.pt')


