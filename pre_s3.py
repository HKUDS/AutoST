
import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString
from shapely.geometry import Polygon,MultiPoint  #多边形
import matplotlib.pyplot as plt
import json
from urllib.request import urlopen, quote
import requests
import geopy
from geopy.geocoders import Nominatim
import copy
import pickle
from datetime import datetime
from itertools import chain
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt

def haversine(lon1, lat1, lon2, lat2): # 
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine公式
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # 地球平均半径，单位为公里
    return c * r * 1000

def load_data(file):
        data_load_file = []
        file_1 = open(file, "rb")
        data_load_file = pickle.load(file_1)
        return data_load_file
    
region_back = load_data("../data/region_back_merge.pickle")
region_traffic = load_data("../data/NY_traffic_2.pickle")


from collections import Counter

sp_tm = []
for item in region_traffic[:]: #['VendorID', 'lpep_pickup_datetime', 'Lpep_dropoff_datetime', 'Store_and_fwd_flag', 'RateCodeID', 'Pickup_longitude', 'Pickup_latitude', 'Dropoff_longitude', 'Dropoff_latitude', 'Passenger_count', 'Trip_distance', 'Fare_amount', 'Extra', 'MTA_tax', 'Tip_amount', 'Tolls_amount', 'Ehail_fee', 'improvement_surcharge', 'Total_amount', 'Payment_type', 'Trip_type ', 'PULocationID', 'DOLocationID']
# for key,value in region_back.items(): ## remember to test the whether the region is [].
    # print(item)
    # println()
    # Point(4, 4)
    dropoff_pos =  Point(item[7],item[8])
    pickup_pos =  Point(item[5],item[6])
    # print("11:",dropoff_pos)
    # print("22:",pickup_pos)
    # poritnlnn()
    tmp_idx = []
    for key,value in region_back.items(): ## remember to test the whether the region is [].
    # for item in region_traffic[:]: #['VendorID', 'lpep_pickup_datetime', 'Lpep_dropoff_datetime', 'Store_and_fwd_flag', 'RateCodeID', 'Pickup_longitude', 'Pickup_latitude', 'Dropoff_longitude', 'Dropoff_latitude', 'Passenger_count', 'Trip_distance', 'Fare_amount', 'Extra', 'MTA_tax', 'Tip_amount', 'Tolls_amount', 'Ehail_fee', 'improvement_surcharge', 'Total_amount', 'Payment_type', 'Trip_type ', 'PULocationID', 'DOLocationID']

        tmp_poly = value
        # poly_shape.intersects(point))
        if dropoff_pos.intersects(tmp_poly):
                dropoff_idx = key
                tmp_idx.append(dropoff_idx)
        if pickup_pos.intersects(tmp_poly):
                pickup_idx = key
                tmp_idx.append(pickup_idx)
    # print("tmp_idx:", tmp_idx)
    # print("item:", item)
    if len(tmp_idx)==2:
        # print("tmp_idx:", tmp_idx)
        # print("item:", item)
        sp_tm.append((tmp_idx[1], tmp_idx[0], item[-1])) #起点/终点/日期
result = pd.value_counts(sp_tm)
print("result:", result)
# println()

unique_region = list(set(sp_tm))

##building flow graph
flow_edges = []
for key,value in result.to_dict().items():
    # print("key:", key)
    # print("value:", value)
    # if value>10:
    # println()
    #pair = ('r_{}_{}'.format(region_dict[key[0]], key[-1]), 'r_{}_{}'.format(region_dict[key[1]], key[-1] + 1), value)
    pair = ('r_{}_{}'.format(key[0], int(key[-1])),'r_{}_{}'.format(key[1], int(key[-1]+1)), {"weight":1, "date":int(key[-1]), "start":'r_{}_{}'.format(key[0], int(key[-1])), "end":'r_{}_{}'.format(key[1], int(key[-1]+1))})
    flow_edges.append(pair)
    # else:
    #     # println()
    #     #pair = ('r_{}_{}'.format(region_dict[key[0]], key[-1]), 'r_{}_{}'.format(region_dict[key[1]], key[-1] + 1), value)
    #     pair = ('r_{}_{}'.format(key[0], int(key[-1])),'r_{}_{}'.format(key[1], int(key[-1]+1)), {"weight":0, "date":int(key[-1]), "start":'r_{}_{}'.format(key[0], int(key[-1])), "end":'r_{}_{}'.format(key[1], int(key[-1]+1))})
    #     flow_edges.append(pair)

print("finish flow graph")

# G_flow = nx.Graph()
# G_flow.add_edges_from(flow_edges[:])

# file=open(r"../data/flow_graph_new_baseline.pickle","wb")
# pickle.dump(G_flow,file) #storing_list
# file.close()
# println()
 

##bulding spatial graph
spatial_dis = []
spatial_dict = {}

flow_nodes = []
for item in unique_region:
    n_1 = "r"+"_"+str(item[0])+"_"+str(item[-1])
    n_2 = "r"+"_"+str(item[1])+"_"+str(int(item[-1])+1)
    if n_1 not in flow_nodes:
        flow_nodes.append(n_1)
    if n_2 not in flow_nodes:
        flow_nodes.append(n_2)

print("finish flow nodes")      
spatial_dis.sort(reverse = False)
spatial_edges = []
spatial_edges.extend(flow_edges) # add edges in flow graph
sim_num=0
for ii in range(len(flow_nodes)):
    for jj in range(ii+1, len(flow_nodes)):
        # time = flow_nodes[ii].split("_")[2]
        t_1 = flow_nodes[ii].split("_")
        t_2 = flow_nodes[jj].split("_")
        # print("t_1:",t_1)
        # print("t_2:",t_2)
        # t_1_pos = np.average(list(zip(*region_back[int(t_1[1])].exterior.coords.xy)), axis = 0)
        # t_2_pos = np.average(list(zip(*region_back[int(t_2[1])].exterior.coords.xy)),axis = 0)
        # t_1_pos = np.average(list(region_back[int(t_1[1])].exterior.coords), axis = 0)
        # t_2_pos = np.average(list(region_back[int(t_2[1])].exterior.coords), axis = 0)
        t_1_pos  = list(region_back[int(t_1[1])].centroid.coords)[0]
        t_2_pos = list(region_back[int(t_2[1])].centroid.coords)[0]
        # print("--:",t_1_pos)
        # print("$$:", t_2_pos)
        # println()

        value = haversine(t_1_pos[0], t_1_pos[1], t_2_pos[0], t_2_pos[1])
        # print("value:", value)
        if value<= 5000:  #小于3公里
            # print("value:",value)
            sim_num+=1
            # yy = key[0].split("_")
            # yy_1 = key[1].split("_")
            # print("key:", key)7000
            # println()
            # print(flow_nodes[ii],flow_nodes[jj])
            # println()
            pair = (flow_nodes[ii],flow_nodes[jj], {"weight":value, "date":int(t_1[2]), "start":flow_nodes[ii], "end":flow_nodes[jj]})
            if pair not in spatial_edges:
                spatial_edges.append(pair)
# print("sim_num:",sim_num)
# print("finish spatial graph--part 2")
# println()

#增加边
params_resolution  = 2
for z in region_back.keys():
    for j in range(params_resolution):
        ox = "r_{}_{}".format(z, j)
        oy = "r_{}_{}".format(z, j+1)
        pair = (ox,oy, {"weight":0, "date":int(j), "start":ox, "end":oy})
        if pair not in spatial_edges:
            spatial_edges.append(pair)
print(len(spatial_edges))
print("finish spatial graph")

G_flow = nx.Graph()
G_flow.add_edges_from(flow_edges[:])
# nx.draw(G_flow, with_labels=True)
# plt.show()
print("G_flow:",G_flow)
#spatial graph
G_spatial = nx.Graph()
G_spatial.add_edges_from(spatial_edges[:])
# nx.draw(G_spatial, with_labels=True)
# plt.show()
print("G_spatial:",G_spatial)

file=open(r"../data/flow_graph_new_1.pickle","wb")
pickle.dump(G_flow,file) #storing_list
file.close()

file=open(r"../data/spatial_graph_new_1.pickle","wb")
pickle.dump(G_spatial,file) #storing_list
file.close()

print("----spatial----:", G_spatial)
print("----flow----:",G_flow)

    