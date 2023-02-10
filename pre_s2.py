
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
taxi = pd.read_csv("../data/2016_Green_Taxi_Trip_Data.csv", sep = ',')
# taxi = pd.read_csv("../data/2015_Green_Taxi_Trip_Data.csv", sep = ',')
# print(taxi[:100])
print(taxi.columns.values.tolist()) #['VendorID', 'lpep_pickup_datetime', 'Lpep_dropoff_datetime', 'Store_and_fwd_flag', 'RateCodeID', 'Pickup_longitude', 'Pickup_latitude', 'Dropoff_longitude', 'Dropoff_latitude', 'Passenger_count', 'Trip_distance', 'Fare_amount', 'Extra', 'MTA_tax', 'Tip_amount', 'Tolls_amount', 'Ehail_fee', 'improvement_surcharge', 'Total_amount', 'Payment_type', 'Trip_type ', 'PULocationID', 'DOLocationID']

def load_data(file):
        data_load_file = []
        file_1 = open(file, "rb")
        data_load_file = pickle.load(file_1)
        return data_load_file
    
# region = load_data("../data/NY_region.pickle")
# selection_dataset['year'] = selection_dataset['Trip Start Timestamp'].map(lambda x: x.split('-')[0])
taxi['date'] = taxi['lpep_pickup_datetime'].map(lambda x:x.split(' ')[0])
# taxi['date'] = taxi['pickup_datetime'].map(lambda x:x.split(' ')[0])
taxi['day'] = taxi['date'].map(lambda x:x.split('/')[1]).apply(int)
taxi["date"] = pd.to_datetime(taxi["date"]).dt.date
s_date = datetime.strptime('20160101', '%Y%m%d').date()
e_date = datetime.strptime('20160101', '%Y%m%d').date()
week_df = taxi[(taxi['date'] >= s_date) & (taxi['date'] <= e_date)]
month_traffic = week_df.drop(['date'], axis=1)
# println()
# month_traffic = y_traffic.loc[[y_traffic['month'] == MONTH]]
#a whole year include 77 regions and a month inlucde 70 regions
#month_traffic = year_traffic
print("one week data:", len(month_traffic))
print(month_traffic['day'])

# pritnln()
month_traffic = month_traffic.values.tolist()
file=open(r"../data/NY_traffic_1_.pickle","wb")
pickle.dump(month_traffic,file) #storing_list
file.close()

