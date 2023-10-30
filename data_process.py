import os

import networkx as nx
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


def data_merge():
    node_data = pd.read_csv('./dataset/train_90.csv')
    edge_data = pd.read_csv('./dataset/edge_90.csv')

    test_node_data = pd.read_csv('./dataset/node_test_4_A.csv')
    test_edge_data = pd.read_csv('./dataset/edge_test_4_A.csv')

    node_data['F_1_EDGE'] = 0
    node_data['F_2_EDGE'] = 0

    # 将edge_data中的F_1按geohash6_point1和date_id分组并求和
    edge_data_grouped_F1 = edge_data.groupby(['geohash6_point1', 'date_id'])['F_1'].sum().reset_index()

    # 将edge_data中的F_2按geohash6_point2和date_id分组并求和
    edge_data_grouped_F2 = edge_data.groupby(['geohash6_point2', 'date_id'])['F_2'].sum().reset_index()

    # 在node_data中匹配并添加F_1_EDGE和F_2_EDGE
    node_data = node_data.merge(edge_data_grouped_F1,
                                how='left',
                                left_on=['geohash_id', 'date_id'],
                                right_on=['geohash6_point1', 'date_id'],
                                suffixes=('', '_new'))
    node_data['F_1_EDGE'] = node_data['F_1_new'].fillna(0)
    node_data.drop(columns=['geohash6_point1', 'F_1_new'], inplace=True)

    node_data = node_data.merge(edge_data_grouped_F2,
                                how='left',
                                left_on=['geohash_id', 'date_id'],
                                right_on=['geohash6_point2', 'date_id'],
                                suffixes=('', '_new'))
    node_data['F_2_EDGE'] = node_data['F_2_new'].fillna(0)
    node_data.drop(columns=['geohash6_point2', 'F_2_new'], inplace=True)

    columns = list(node_data.columns)
    columns.insert(-4, 'F_1_EDGE')
    columns.insert(-4, 'F_2_EDGE')
    node_data = node_data[columns]

    node_data = node_data.iloc[:, :-2]

    # 归一化F_2_EDGE，F_1_EDGE
    scaler = MinMaxScaler()
    node_data[['F_1_EDGE', 'F_2_EDGE']] = scaler.fit_transform(node_data[['F_1_EDGE', 'F_2_EDGE']])

    # 保存结果
    node_data.to_csv('./dataset/train_with_edge_data.csv', index=False)

    test_node_data['F_1_EDGE'] = 0
    test_node_data['F_2_EDGE'] = 0

    # 将edge_data中的F_1按geohash6_point1和date_id分组并求和
    test_edge_data_grouped_F1 = test_edge_data.groupby(['geohash6_point1', 'date_id'])['F_1'].sum().reset_index()
    test_edge_data_grouped_F2 = test_edge_data.groupby(['geohash6_point2', 'date_id'])['F_2'].sum().reset_index()

    # 在node_data中匹配并添加F_1_EDGE和F_2_EDGE
    test_node_data = test_node_data.merge(test_edge_data_grouped_F1,
                                          how='left',
                                          left_on=['geohash_id', 'date_id'],
                                          right_on=['geohash6_point1', 'date_id'],
                                          suffixes=('', '_new'))

    test_node_data['F_1_EDGE'] = test_node_data['F_1_new'].fillna(0)
    test_node_data.drop(columns=['geohash6_point1', 'F_1_new'], inplace=True)

    test_node_data = test_node_data.merge(test_edge_data_grouped_F2,
                                            how='left',
                                            left_on=['geohash_id', 'date_id'],
                                            right_on=['geohash6_point2', 'date_id'],
                                            suffixes=('', '_new'))

    test_node_data['F_2_EDGE'] = test_node_data['F_2_new'].fillna(0)
    test_node_data.drop(columns=['geohash6_point2', 'F_2_new'], inplace=True)

    # 归一化F_2_EDGE，F_1_EDGE
    scaler1 = MinMaxScaler()
    test_node_data[['F_1_EDGE', 'F_2_EDGE']] = scaler1.fit_transform(test_node_data[['F_1_EDGE', 'F_2_EDGE']])

    test_node_data.to_csv('./dataset/test_with_edge_data.csv', index=False)


def data_process():
    if not os.path.exists('./dataset/train_with_edge_data.csv'):
        data_merge()
    node_data = pd.read_csv('./dataset/train_with_edge_data.csv')

    num_groups = len(node_data) // 45

    # 初始化训练集和测试集
    train_data = pd.DataFrame()
    valid_data = pd.DataFrame()

    # 按照每组45行的方式分割数据
    for i in range(num_groups):
        start = i * 90
        end = start + 80  # 前5行用于训练
        train_data = pd.concat([train_data, node_data.iloc[start:end]])

        start = end
        end = start + 10  # 后5行用于测试
        valid_data = pd.concat([valid_data, node_data.iloc[start:end]])

    return train_data, valid_data


def test_data_process():
    node_data = pd.read_csv('./dataset/test_with_edge_data.csv')

    return node_data


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


if "__main__" == __name__:
    train_data, valid_data = data_process()
