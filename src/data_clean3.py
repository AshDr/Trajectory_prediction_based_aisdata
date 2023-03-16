import copy
import math
import os.path
import numpy as np
import pandas as pd
from pandas import Series
from pandas.core.frame import DataFrame
from pyproj import Transformer

# pd.set_option('display.max_rows', 50)
# pd.set_option('display.min_rows', 20)
# pd.set_option('display.max_columns', 10)

max_chabu_delta = 10  # 插补时间间隔 > 10min的数据

# features = ['MMSI', 'BaseDateTime', 'X', 'Y', 'SOG', 'COG', 'Heading', 'Status']

features = ['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'COG', 'Heading', 'Status']
# 插值处理


def chabu(df: DataFrame) -> DataFrame:
    global chabucnt
    new_lst = []
    num = 0
    for i, row in enumerate(df.itertuples(index=False)):
        curtime, lat, lon = row[1], row[2], row[3]
        x, y = transformer.transform(lat, lon)
        if len(new_lst) > 0 and (curtime - new_lst[-1][1]).seconds > max_chabu_delta * 60:
            delta = (row[1] - new_lst[-1][1]).seconds // (max_chabu_delta * 60)
            chabucnt += delta
            for j in range(delta):
                new_lst.append([row[0], np.nan, np.nan, np.nan, np.nan, np.nan, row[-2], row[-1]])
        new_lst.append(df.values[i])
        # new_lst[-1][2], new_lst[-1][3] = x, y     # 不用x 和 y 直接用经纬度

    newdf = pd.DataFrame(new_lst, columns=df.columns)
    for col in df.columns:
        if col == 'BaseDateTime' or col == 'MMSI' or col == 'Heading' or col == 'Status':
            continue
        s = Series(data=newdf[col], dtype='float64')
        s.interpolate(method='pchip', inplace=True)
        newdf[col] = s
    t = pd.to_numeric(newdf['BaseDateTime'])
    t[t < 0] = np.nan
    t = t.interpolate()
    newdf['BaseDateTime'] = pd.to_datetime(t)
    newdf.columns = features
    return newdf


def get_speed(x, y, t, prex, prey, pret):
    dis = math.sqrt((x - prex) * (x - prex) + (y - prey) * (y - prey))  # 米
    tdelta = (t - pret).seconds  # 秒
    return dis / tdelta  # 米每秒


transformer = Transformer.from_crs("epsg:4326", "epsg:32649")  # WGS 转 UTM坐标
VMAX = 25 * 1852 / 3600  # max  m/s


# print(VMAX)

# 数据预处理
def work(df: DataFrame) -> DataFrame:
    """
    先做异常值丢弃(轨迹点偏离)
    不合理行为删除
    再做插补
    :param df:
    :return:
    """
    print("------")
    # print(df)
    prelat = ""
    prelon = ""
    pretime = ""
    for i, row in enumerate(df.itertuples(index=False)):
        curtime, lat, lon, sog = row[1], row[2], row[3], row[4]
        x, y = transformer.transform(lat, lon)
        if i > 0:
            prex, prey = transformer.transform(prelat, prelon)
            if math.isclose(lat, prelat) and math.isclose(lon, prelon) and \
                    (not math.isclose(sog, 0)):  # 异常数据
                df.drop(i, inplace=True)
            elif get_speed(x, y, curtime, prex, prey, pretime) > VMAX:  # 速度 > vmax 就是异常点
                df.drop(i, inplace=True)
            else:
                pretime, prelat, prelon = row[1], row[2], row[3]
        else:
            pretime, prelat, prelon = row[1], row[2], row[3]
    res = chabu(df)  # 进行插补
    return res


TMPS = []

Q = []

for i in range(1000):  # 一千条船
    TMPS.clear()
    Q.clear()
    path = '../data/boat_data/data_' + str(i) + '.csv'  # 拿到一个船的所有信息
    df = pd.read_csv(path)
    df = df.sort_values(by=['BaseDateTime'])  # 有的时间点没有按顺序排 要在这里排序一遍
    df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'])

    # print(df)

    for row in df.itertuples(index=False):
        if len(TMPS) == 0:
            TMPS.append(row)
        else:
            sog = row[4]
            TMPS.append(row)
            if math.isclose(sog, 0):  # 船速为0作为分割点
                if len(TMPS) >= 20:
                    Q.append(copy.deepcopy(TMPS))
                TMPS.clear()
            elif (row[1] - TMPS[-1][1]).seconds >= 2 * 3600:  # 间隔时间超过2h认为是两条轨迹
                if len(TMPS) >= 20:  # 数据长度限制
                    Q.append(copy.deepcopy(TMPS))
                TMPS.clear()
    if len(TMPS) >= 20:  # 轨迹点至少要有20个
        Q.append(copy.deepcopy(TMPS))

    """
     这里开始要做数据的预处理
     1.异常值丢弃
     2.航行中存在不合理行为就要对数据进行删除
     3.如果两个点之间的时间间隔大于y 就要进行插补 (重点)
    """

    dir_path = '../data/path_data/id' + str(i) + '/'
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    chabucnt = 0
    for j, lst in enumerate(Q):
        new_df = work(DataFrame(lst))
        # print(new_df.describe())
        # print(new_df)
        save_path = '../data/path_data/id' + str(i) + '/path_' + str(j) + '.csv'
        new_df.to_csv(save_path, index=False)
    print(chabucnt)
