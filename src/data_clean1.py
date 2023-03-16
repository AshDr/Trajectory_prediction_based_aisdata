import copy

import torch
import pandas as pd
import numpy as np

# pd.set_option('display.max_columns', None)
reader = pd.read_csv('../data/AIS_2019_01_01.csv', sep=',', chunksize=50000)  # 分块
for i, chunk in enumerate(reader):
    df1 = chunk.iloc[:, 0:7]
    df2 = chunk.iloc[:, 11:12]
    res = pd.concat([df1, df2], axis=1)
    res.to_csv('../data/rawdata_'+str(i)+'.csv', index=False)


