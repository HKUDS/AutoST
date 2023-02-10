
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
import time
# taxi = pd.read_csv("../data/2016_Green_Taxi_Trip_Data.csv", sep = ',')
# print(taxi[:2])

census_block = pd.read_excel("../data/rollingsales_manhattan.xlsx",skiprows = 4)
# print(census_block[:2])
print(census_block.columns.values.tolist())
blocks = copy.deepcopy(census_block).values.tolist()


# region = census_block["BUILDING CLASS CATEGORY"].values.tolist()
region = census_block["BUILDING CLASS AT TIME OF SALE"].values.tolist()


region_ = list(set(region))
reg_nyc_dict = {} ##113 region in manhattan
for idx,sub in enumerate(region_):
    reg_nyc_dict[sub] = idx
# print(reg_nyc_dict)
# print(len(reg_nyc_dict))
# println()

skip_num = 0
region_f = {}
add_pos = {}
i= 0
NYC_house_middle = []
for sline in blocks:
    start_t = time.time()
    i+=1
    tmp = []
    # print("sline:", sline[8],sline[18],sline[14], sline[19])
    # print("address:",sline[8])
    t = sline[8].split(",")
    ##collect lat,lon
    geolocater = Nominatim(user_agent='demo_of_gnss_help')
    try:
        if t[0] not in add_pos.keys():
            # print("not in here")
            location = geolocater.geocode(t[0])
            if hasattr(location,'latitude') and (location.latitude is not None) and hasattr(location,'longitude') and (location.longitude is not None):
                # print([location.latitude, location.longitude])
                # println()
                # print("t:", t)
                # tmp.append([location.latitude, location.longitude])
                # tmp.append(reg_nyc_dict[sline[18]])
                add_pos[t[0]] = [location.latitude, location.longitude]
                tmp.append(reg_nyc_dict[sline[18]])
                tmp.append(sline[14])
                tmp.append(sline[19])
                # print("--:",float(sline[19])/float(sline[14]))
                tmp.append(float(sline[19]))
                if reg_nyc_dict[sline[18]] not in region_f.keys():
                    region_f[reg_nyc_dict[sline[18]]] = []
                    region_f[reg_nyc_dict[sline[18]]].append([location.latitude, location.longitude])
                else:
                    region_f[reg_nyc_dict[sline[18]]].append([location.latitude, location.longitude])
                NYC_house_middle.append(tmp)
                
        else:
            # print("---in here---")
            # print("add_pos[t[0]]:", add_pos[t[0]])
            tmp.append(reg_nyc_dict[sline[18]])
            tmp.append(sline[14])
            tmp.append(sline[19])
            # print("--:",float(sline[19])/float(sline[14]))
            # tmp.append(float(sline[19]))
            if reg_nyc_dict[sline[18]] not in region_f.keys():
                region_f[reg_nyc_dict[sline[18]]] = []
                region_f[reg_nyc_dict[sline[18]]].append(add_pos[t[0]])
            else:
                region_f[reg_nyc_dict[sline[18]]].append(add_pos[t[0]])
            NYC_house_middle.append(tmp)
    except IOError:
        add_pos[t[0]] = []
        skip_num+=1
        # print('skip this row')
    print("i:", i)
    print(time.time()-start_t)

print(region_f)    
print(NYC_house_middle[:3])
print(len(NYC_house_middle))
print(len(region_f))
print(len(add_pos))
print("skip_num",skip_num)

file=open(r"../data/NY_house.pickle","wb")
pickle.dump(NYC_house_middle,file) #storing_list
file.close()
file=open(r"../data/NY_stree_pos.pickle","wb")
pickle.dump(add_pos,file) #storing_list
file.close()

file=open(r"../data/NY_region.pickle","wb")
pickle.dump(region_f,file) #storing_list
file.close()

