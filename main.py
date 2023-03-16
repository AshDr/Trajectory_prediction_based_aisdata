import copy

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pandas.core.frame import DataFrame


class Net(nn.Module):
    def __init__(self, feature_size, size_hidden, dropout=0):
        super().__init__()
        self.feature_size = feature_size
        self.size_hidden = size_hidden
        self.rnn = nn.LSTM(input_size=feature_size, hidden_size=size_hidden, dropout=dropout, batch_first=True)

    def forward(self, input):
        output, _ = self.rnn(input)
        return output


useful_tag = ['X', 'Y', 'SOG', 'COG', 'Heading']
predict_tag = ['X', 'Y']
len_topredict = 10


class MyDataSet(Dataset):
    def __init__(self, df: DataFrame):
        self.x_data = df[useful_tag].astype('float32').values
        self.y_data = df[predict_tag].astype('float32').values
        self.length = len(self.x_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.length


def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
            print(f'epoch: {epoch + 1}', f'loss')


features = ['MMSI', 'BaseDateTime', 'X', 'Y', 'SOG', 'COG', 'Heading', 'Status']
data_file_root_path = './data/path_data/'
tmp_lst = []
for i in range(10):
    data_file_path = data_file_root_path + 'id' + str(i) + '/'

    """
        还要做一步归一化
    """
    file_lst = os.listdir(data_file_path)
    for filename in file_lst:
        path = data_file_path + filename
        df = pd.read_csv(path)
        if len(tmp_lst)<1000:
            dataset = MyDataSet(df)
            for j in range(len_topredict, len(dataset)):
                X, y = dataset.x_data[j - len_topredict:j], dataset.y_data[j - len_topredict:j]
                tmp_lst.append((X, y))
            # train_loader = DataLoader(dataset=dataset, batch_size=3)  # 这里batch_size就是一次从一个轨迹中取几个点

for j, data in enumerate(tmp_lst):
    inputs, labels = data

ratio = 0.8
loss = nn.MSELoss(reduction='none')
net = Net(4, 10, 0)
n = torch.rand(12, 4)
out = net(n)  # [seqlen, hidden_size]

# 20个点轨迹 shape =  [20, 5]
# 目标后面一个时间的x,y值 shape = [1, 2]
