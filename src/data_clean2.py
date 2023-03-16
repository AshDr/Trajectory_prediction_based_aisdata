import copy
import math
import os.path

import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np

S = []  # 每个船只存放它的所有点 list[list[tuple(),tuple(),...],list[],...]
mydict = {}  # 映射每个船只编号
cnt = 0
mx = 0
for i in range(151):
    file_path = '../data/rawdata_'+str(i)+'.csv'
    df = pd.read_csv(file_path)
    for row in df.itertuples(index=False):
        mmsi = row[0]
        if mmsi not in mydict:
            if cnt == 1000:    # 最多1000条船的信息
                continue
            mydict[mmsi] = cnt
            cnt += 1
            S.append(list())
        idx = mydict[mmsi]
        S[idx].append(row)
    for j in range(cnt):
        save_path = '../data/boat_data/data_'+str(j)+'.csv'
        newdf = DataFrame(S[j])
        if not os.path.exists(save_path):
            newdf.to_csv(save_path, index=False)
        else:
            newdf.to_csv(save_path, mode='a', index=False, header=False)
        S[j].clear()


