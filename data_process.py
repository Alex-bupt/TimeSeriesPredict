import networkx as nx
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


def data_process():
    train_data = pd.read_csv('./dataset/train_90.csv')
    edge_data = pd.read_csv('./dataset/edge_90.csv')

    train_features = train_data.columns[2:-2]
    edge_features = edge_data.columns[2:-2]

    # 创建一个MinMaxScaler对象
    scaler = MinMaxScaler()

    # 使用fit_transform方法对选定的特征列进行归一化
    train_data[train_features] = scaler.fit_transform(train_data[train_features])
    edge_data[edge_features] = scaler.fit_transform(edge_data[edge_features])

    return train_data, edge_data


def graph_build(train_data, edge_data):
    # 创建一个空的图
    G = nx.Graph()

    # 添加节点和属性特征
    for index, row in train_data.iterrows():
        geohash_id = row['geohash_id']
        G.add_node(geohash_id)

    # 添加边和属性特征
    for index, row in edge_data.iterrows():
        geohash1 = row['geohash6_point1']
        geohash2 = row['geohash6_point2']
        features = row[2:-1]  # 去除geohash6_point1和geohash6_point2以及date_id
        date_id = row['date_id']
        G.add_edge(geohash1, geohash2, features=features, date_id=date_id)

    return G


# 创建一个自定义的数据集类
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
