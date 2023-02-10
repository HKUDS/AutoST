
import pickle
import pandas as pd
from itertools import chain
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt


def load_data(file):
    data_load_file = []
    file_1 = open(file, "rb")
    data_load_file = pickle.load(file_1)
    return data_load_file
from scipy.sparse import csr_matrix

flow_g = load_data('../data/flow_graph_2.pickle')

spatial_g = load_data('../data/spatial_graph_new_1.pickle')
region_attr_g = load_data('../data/region_attr_graph_test.pickle')


flow_nodes = list(flow_g.nodes)
spatial_nodes = list(spatial_g.nodes)
# flowsum_nodes = list(flow_sum_g.nodes)
regat_nodes = list(region_attr_g.nodes)
flow_edges = list(flow_g.edges(data=True))
# print("****:",len(flowsum_nodes))
# println()
spatial_edges = list(spatial_g.edges(data=True))
# print(":**************",spatial_edges)
# println()
# flowsum_edges = list(flow_sum_g.edges(data=True))
regat_edges = list(region_attr_g.edges(data=True))
# print(regat_edges)


part_f = flow_nodes
part_s = spatial_nodes
# part_flow = flowsum_nodes


hy_edges = []
for sub in regat_nodes:
    for ss in flow_nodes:
        tmp_ss = ss.split("_")
        tmp_sub = sub.split("_")
        tmp_c = tmp_ss[0]+'_'+tmp_ss[1]
        tmp_s = tmp_sub[0]+'_'+tmp_sub[1]
       
        if tmp_s == tmp_c:
            # pair = (sub, ss,{"weight":1, "date": tmp[2], "start":sub, "end":ss})
            pair = (sub, ss,{"weight":1, "date": tmp_ss[2], "start":sub, "end":ss})
            # print("pair:", pair)
            # println()
            # if pair not in hy_edges:
            hy_edges.append(pair)
print(len(hy_edges))

for ss in spatial_nodes:
    for ff in flow_nodes:
        tps = ss.split("_")
        # tps_c = tps[0]+'_'+tps[1]
        tpf = ff.split("_")
        # tpf_c = tpf[0]+'_'+tpf[1]
        # print("ff:",ff)
        # print("ss:",ss)
        # print(tpf)
        # println()
        # ss_=ss+"_"+"s"
        if tps[1] == tpf[1]:
            # pair = (ss, ff,{"weight":0, "date":tpf[2] , "start":ss, "end":ff})
            pair = (ss, ff,{"weight":0, "date":1 , "start":ss, "end":ff})
            # print("pair:", pair)
            # pritnln()
            hy_edges.append(pair)
      

print("hy_edges:",len(hy_edges))

G_hy = nx.Graph()
G_hy.add_edges_from(hy_edges)
G_hy.add_edges_from(flow_edges)
# G_hy.add_edges_from(flowsum_edges)
G_hy.add_edges_from(spatial_edges)
G_hy.add_edges_from(regat_edges)
# nx.draw(G_hy)
# plt.show()
print("hyper_grapgh:", G_hy)

print(G_hy) 
nodes_num = 3
file=open(r"../data/hy_new_test_60.pickle","wb")
pickle.dump(G_hy,file) #storing_list
file.close()

