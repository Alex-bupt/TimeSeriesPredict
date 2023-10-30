import networkx as nx
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset




# 将id映射成数字然后归一化################################



def data_process():
    node_data = pd.read_csv('./dataset/train_90.csv')
    edge_data = pd.read_csv('./dataset/edge_90.csv')

    # 将id映射成数字然后归一化
    unique_geohash_ids = node_data['geohash_id'].unique()
    geohash_id_to_number = {geohash_id: i for i, geohash_id in enumerate(unique_geohash_ids)}

    node_data['geohash_id'] = node_data['geohash_id'].map(geohash_id_to_number)
    edge_data['geohash6_point1'] = edge_data['geohash6_point1'].map(geohash_id_to_number)
    edge_data['geohash6_point2'] = edge_data['geohash6_point2'].map(geohash_id_to_number)

    # train_features = node_data.columns[2:-2]
    # edge_features = edge_data.columns[2:-2]
    #
    # # 创建一个MinMaxScaler对象
    # scaler = MinMaxScaler()
    #
    # # 使用fit_transform方法对选定的特征列进行归一化
    # node_data[train_features] = scaler.fit_transform(node_data[train_features])
    # edge_data[edge_features] = scaler.fit_transform(edge_data[edge_features])

    # 切分数据集
    cnt = int(len(node_data) / 30)

    # 将cnt*0.8取整作为训练集，剩下的作为验证集
    train_data = node_data[:int(cnt * 0.9) * 30]
    valid_data = node_data[int(cnt * 0.9) * 30:]

    return train_data, valid_data, edge_data


def test_data_process():
    node_data = pd.read_csv('./dataset/node_test_4_A.csv')
    edge_data = pd.read_csv('./dataset/edge_test_4_A.csv')

    unique_geohash_ids = node_data['geohash_id'].unique()
    geohash_id_to_number = {geohash_id: i for i, geohash_id in enumerate(unique_geohash_ids)}

    node_data['geohash_id'] = node_data['geohash_id'].map(geohash_id_to_number)
    edge_data['geohash6_point1'] = edge_data['geohash6_point1'].map(geohash_id_to_number)
    edge_data['geohash6_point2'] = edge_data['geohash6_point2'].map(geohash_id_to_number)

    # train_features = node_data.columns[2:]
    # edge_features = edge_data.columns[2:-2]
    #
    # # 创建一个MinMaxScaler对象
    # scaler = MinMaxScaler()
    #
    # # 使用fit_transform方法对选定的特征列进行归一化
    # node_data[train_features] = scaler.fit_transform(node_data[train_features])
    # edge_data[edge_features] = scaler.fit_transform(edge_data[edge_features])

    return node_data, edge_data


# 构建权重矩阵
def build_weight_matrix():
    node_data = pd.read_csv('./dataset/train_90.csv')
    edge_data = pd.read_csv('./dataset/edge_90.csv')

    unique_geohash_ids = node_data['geohash_id'].unique()
    geohash_id_to_number = {geohash_id: i for i, geohash_id in enumerate(unique_geohash_ids)}

    edge_data['geohash6_point1'] = edge_data['geohash6_point1'].map(geohash_id_to_number)
    edge_data['geohash6_point2'] = edge_data['geohash6_point2'].map(geohash_id_to_number)

    # 构建权重矩阵


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


if "__name__" == "__main__":
    train_data, valid_data, edge_data = data_process()
    test_node_data, test_edge_data = test_data_process()
    print(train_data.head())
    print(valid_data.head())
    print(edge_data.head())
    print(test_node_data.head())
    print(test_edge_data.head())

