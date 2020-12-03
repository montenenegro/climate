# -*- coding: utf-8 -*-
"""

@author: Adrien Wehrlé, GEUS (Geological Survey of Denmark and Greenland)

"""

import glob
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

stations_list = glob.glob('C:/Users/Pascal/Desktop/GEUS_2019/GISTEMP_analysis/'
                          + 'raw_GISTEMP_csv_data/*csv')

valid_files = 0

for i, station in tqdm(enumerate(stations_list)):
    
    data = pd.read_csv(station, na_values=999.9, index_col=0).iloc[:, :12]
    dataset = data.stack(dropna=False).reset_index()
    dataset.rename(columns={'level_1': 'MONTH', 0: 'Temperature_C'}, inplace=True)
    dataset.index = pd.to_datetime(dataset.YEAR.astype(str) + dataset.MONTH, format='%Y%b')
    dataset = dataset[((dataset.YEAR >= 1979) & (dataset.YEAR <= 2020))]
    dataset.drop(['YEAR', 'MONTH'], axis=1, inplace=True)
    missing_values_perc = np.sum(np.isnan(dataset))[0] / len(dataset)
    
    if missing_values_perc < 0.25:
        
        dataset.to_csv('C:/Users/Pascal/Desktop/UGAM2/CIA/climatic-modes-arctic/'
                       + 'raw_GISTEMP_csv_data/' + station.split(os.sep)[-1])
        valid_files += 1


# pd.read_csv('file:///C:/Users/Pascal/Desktop/UGAM2/CIA/climatic-modes-arctic/raw_GISTEMP_csv_data/ASN00002012.csv', index_col=0)
