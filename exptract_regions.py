
import pandas as pd
from shapely.geometry import Point, LineString
from shapely.geometry import Polygon,MultiPoint,MultiPolygon  
import numpy as np
import json
import geopandas
import shapefile

 
m_region =[]
shp_df = geopandas.GeoDataFrame.from_file("../data/2010 Census Blocks/geo_export_c80540b5-38fc-4bb4-81cd-ae8082c49f02.shp",encoding = 'gb18030').values.tolist()
for item in shp_df:
    if item[2] == "Manhattan":
        m_region.append(item)

# print(len(m_region))
q_index = []
for hh in m_region:
    if hh[3] not in q_index:
        q_index.append(hh[3])
print(len(q_index))
region_dict = {}
for item in m_region:
    if item[3] not in region_dict.keys():
        region_dict[item[3]] = item
    else:
        if item[5] > region_dict[item[3]][5]:
            region_dict[item[3]] = item
# print(m_region)
# print(len(region_dict))
region_trans = {}
for key,value in region_dict.items():
    region_trans[int(key)] = value[-1]
# print(region_trans[1051])
# print(len(region_trans))
region_s = {}
for idx,im in enumerate(region_trans.items()):
    region_s[idx] = im[1]
# print(len(region_s))
# print(region_s[0])
import pickle
file=open(r"../data/region_back.pickle","wb")
pickle.dump(region_s,file) #storing_list
file.close()

    

            
  
        
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
        