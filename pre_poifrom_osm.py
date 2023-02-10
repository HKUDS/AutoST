
import pandas as pd
import pickle
from shapely.geometry import Point, LineString
from shapely.geometry import Polygon,MultiPoint  #多边形
import torch
from torch import nn


def load_data(file):
    data_load_file = []
    file_1 = open(file, "rb")
    data_load_file = pickle.load(file_1)
    return data_load_file
poi = pd.read_csv("../data/poi_nyc.csv",sep=",").values.tolist()
region_back = load_data("../data/region_back_merge.pickle")
# print(poi.columns.values.tolist())
# pritnln()
region_poi={}
poi_list=[]
for key,value in region_back.items():
    region_poi[key] = []
for item in poi:
    # print(item[23], item[84], item[92])
    for key,value in region_back.items():
        tmp_point = Point(item[3],item[0])
        if tmp_point.intersects(value):
            if item[23]!=" ":
                if item[23] not in region_poi[key]:
                    region_poi[key].append(item[23])
                if item[23] not in poi_list:
                    poi_list.append(item[23])
            elif item[84]!=" ":
                if item[84] not in region_poi[key]:
                    region_poi[key].append(item[84])
                if item[84] not in poi_list:
                    poi_list.append(item[84])
print(region_poi)
print(poi_list)
# poi_list = ['drinking_water', 'toilets', 'school', 'hospital', 'arts_centre', 'fire_station', 'police', 'bicycle_parking', 'fountain', 'ferry_terminal', 'bench', 'cinema', 'cafe', 'pub', 'waste_basket', 'parking_entrance', 'parking', 'fast_food', 'bank', 'restaurant', 'ice_cream', 'pharmacy', 'taxi', 'post_box', 'atm', 'nightclub', 'social_facility', 'bar', 'biergarten', 'clock', 'bicycle_rental', 'community_centre', 'watering_place', 'ranger_station', 'boat_rental', 'recycling', 'payment_terminal', 'bicycle_repair_station', 'place_of_worship', 'shelter', 'telephone', 'clinic', 'dentist', 'vending_machine', 'theatre', 'charging_station', 'public_bookcase', 'post_office', 'fuel', 'doctors']
poi_dict = {}
for idx,item in enumerate(poi_list):
    poi_dict[item]=idx
print("sum of the category of POI:", len(poi_dict))
reg_incld_poi={}
for key,value in region_poi.items():
    reg_incld_poi[key] = []
    for uu in value:
        if uu in poi_dict.keys():
            reg_incld_poi[key].append(poi_dict[uu])
print("reg_incld_poi:",reg_incld_poi)

import pickle
file=open(r"../data/reg_incld_poi_new.pickle","wb")
pickle.dump(reg_incld_poi,file) #storing_list
file.close()

file=open(r"../data/poi_dict_new.pickle","wb")
pickle.dump(poi_dict,file) #storing_list
file.close()












