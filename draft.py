import pandas as pd
import numpy as np
from pandas import Series

# path = 'data/boat_data/data_0.csv'
# dtype = {'MMSI': 'int64',
#          'LAT': 'float64',
#          'LON': 'float64',
#          'SOG': 'float64',
#          'COG': 'float64',
#          'Heading': 'float64',
#          'Status': 'float64'}
# df = pd.read_csv(path, dtype=dtype, parse_dates=True)
# df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'])
#
# df = pd.DataFrame(np.insert(df.values, 2, values=[np.nan, np.nan, np.nan, np.nan, np.nan,
#                                                   np.nan, np.nan, np.nan], axis=0), columns=df.columns,
#                   )

# df = pd.DataFrame(data=[[1, 2, 3], [4, 5, 6]], columns=['a', 'b', 'c'])
# for i, row in enumerate(df.iterrows()):
#     print(row[1])
# print(df)
from torch import tensor
from sklearn.utils import shuffle
from sklearn.cluster import DBSCAN
import scipy as sp
import os
from sklearn.preprocessing import MinMaxScaler
data_file_root_path = 'data/path_data/'

for i in range(1):
    data_file_path = data_file_root_path + 'id' + str(i) + '/'
    """
        还要做一步归一化
    """
    file_lst = os.listdir(data_file_path)
    for filename in file_lst:
        path = data_file_path + filename
        df = pd.read_csv(path)
        df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'])
        df['year'] = df['BaseDateTime'].dt.year
        df['month'] = df['BaseDateTime'].dt.month
        df['day'] = df['BaseDateTime'].dt.day
        df['hour'] = df['BaseDateTime'].dt.hour
        new_df = df.iloc[:, 2:]
        tmp = np.array(new_df)
        tmp = MinMaxScaler(tmp)
        print(tmp)
        model = DBSCAN(eps=10, min_samples=3)
        model.fit(new_df)
        print(model.labels_)





def dbscan_predict(dbscan_model, X_new, metric=sp.spatial.distance.euclidean):
    # Result is noise by default
    y_new = np.ones(shape=len(X_new), dtype=int) * -1
    # Iterate all input samples for a label
    for j, x_new in enumerate(X_new):
        # Find a core sample closer than EPS
        for i, x_core in enumerate(dbscan_model.components_):
            if metric(x_new, x_core) < dbscan_model.eps:
                # Assign label of x_core to x_new
                y_new[j] = dbscan_model.labels_[dbscan_model.core_sample_indices_[i]]
                break

    return y_new

# c = np.concatenate((a,b),axis=2)
