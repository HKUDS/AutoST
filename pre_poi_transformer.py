
import pandas as pd
import pickle
from shapely.geometry import Point, LineString
from shapely.geometry import Polygon,MultiPoint  
import torch
from torch import nn
import numpy as np


def load_data(file):
    data_load_file = []
    file_1 = open(file, "rb")
    data_load_file = pickle.load(file_1)
    return data_load_file
region_back = load_data("../data/region_back_merge.pickle")
reg_poi = load_data("../data/reg_incld_poi_new.pickle")
# reg_spatial = load_data("../data/region_spatial.pickle")
poi_max=[]
for key,value in reg_poi.items():
    poi_max.extend(value)
print(max(poi_max)) #there are 120 fin-grained pois
# println()
reg_poi_={}
s = 0
emb = nn.Embedding(120, 512)
embedding_spatial = torch.nn.Embedding(15, 512)  # spatial
for key,value in reg_poi.items():
    # print("value:",value)
    if value!=[]:
        reg_poi_[key]=[]
        # print("value:",value)
        if len(value)>s:
            s = len(value)
        for item in value:
            reg_poi_[key].append(emb(torch.tensor(item)).tolist())
# spa_vec= embedding_spatial(torch.tensor(reg_spatial[idx]))
# reg_poi_t = {}
reg_poi_list = []
for iii in range(180):
# for key,value in reg_poi_.items():
    if iii not in reg_poi_.keys():
        reg_poi_list.append(np.array([0.0]*512))
        # reg_poi_list.append(ci)
    else:
        # print("value:",value)
        tp = np.mean(reg_poi_[key],axis=0)
        reg_poi_list.append(tp)
reg_poi_list_  = torch.tensor(np.array(reg_poi_list)).float()
reg_poi_list_tensor = torch.unsqueeze(reg_poi_list_,0)
print(reg_poi_list_tensor.size())

reg_idx= [key for key in reg_poi_.keys()]
from torch import nn
encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8 )
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
# src = torch.rand(1, 172, 512)
src = reg_poi_list_tensor
out = transformer_encoder(src)
# print(out.size())
out_ = torch.squeeze(out,0)
print(out_.size())
print(reg_idx)
print(len(reg_idx))
# reg_poi_vec = {}
# for idx,vec in zip(reg_idx,out_):
#     reg_poi_vec[idx] = vec

file=open(r"../data/reg_poi_vec.pickle","wb")
pickle.dump(out_,file) #storing_list
file.close()

file=open(r"../data/reg_poi_idx.pickle","wb")
pickle.dump(reg_idx,file) #storing_list
file.close()
        






    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

