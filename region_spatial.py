

import pickle
import numpy as np
# import matplotlib.pyplot as plt
from sklearn import metrics
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
from shapely.geometry import Polygon
from shapely import wkt
import geopandas as gpd
import math
from math import cos
def load_data(file):
        data_load_file = []
        file_1 = open(file, "rb")
        data_load_file = pickle.load(file_1)
        return data_load_file

para= 3000
region_pos = load_data("../data/region_back.pickle")
reg_sm = {}
all_pos = []
for key,value in region_pos.items():
    reg_sm[key] = list(value.centroid.coords)[0]
    all_pos.append(list(value.centroid.coords)[0])
max_lon = max([item[0] for item in all_pos])
min_lon = min([item[0] for item in all_pos])
max_lat = max([item[1] for item in all_pos])
min_lat = min([item[1] for item in all_pos])
# print(max_lon,min_lon,max_lat,min_lat)
dis_lon = (max_lon-min_lon)*111100
lon_num = math.ceil(dis_lon/para)


dis_lat = (max_lat-min_lat)*111100*cos(max_lat-min_lat)
# print(dis_lat)
lat_num = math.ceil(dis_lat/para)
# print(lon_num,lat_num)

reg_token = {}
li=[]
for idx,pos in enumerate(all_pos):
    lon = pos[0]-min_lon
    lat = pos[1]-min_lat
    x,y = int(lon*111100/para),int(lat*111100*cos(lat)/para)
    tok= x*21+y
    if tok not in li:
        li.append(tok)
    reg_token[idx] = tok
    # print("cor_token:", idx,x,y,tok)
# print(reg_token)
print(len(li))
li_map = {}
for idx,uu in enumerate(li):
    li_map[uu] = idx
reg_t_con ={}
ton=[]
for key,value in reg_token.items():
    reg_t_con[key] = li_map[value]
    if li_map[value] not in ton:
        ton.append(li_map[value])
print(reg_t_con)
print(max(ton))

file=open(r"../data/region_spatial.pickle","wb")
pickle.dump(reg_t_con,file) #storing_list
file.close()



