import pandas as pd
import numpy as np

def get_Timeindex(time):
    time=str(time)
    time=time[5:]
    match time[0]:
        case '1':
            time_index=int(time[1:-1])-4
        case '2':
            time_index=int(time[1:-1])+27
        case '3':
            time_index=int(time[1:-1])+55
        case '4':
            time_index=int(time[1:-1])+86
        case _:
            print('TIME INDEX ERROR!')
    return time_index

def save_Index():
    node_data = pd.read_csv('./dataset/node_test_4_A.csv')
    node_num = len(node_data) // 4

    index_dic={}
    for i in range(0,node_num):
        index_dic[i]=node_data.loc[i*4,'geohash_id']
        index_dic[node_data.loc[i*4,'geohash_id']]=i

    # print(type(index_dic))
    np.save('./dataset/index.npy',index_dic)

def get_Index():
    index = np.load('./dataset/index.npy',allow_pickle=True)
    return index.item()

def save_Adjmatrix():
    adj_data = pd.read_csv('./dataset/edge_90.csv')
    adj_mat=np.zeros((90,1140,1140,2),dtype=int) # T*N*N*C

    index=get_Index()
    for i in range(0,len(adj_data)):
        if index.get(adj_data.loc[i,'geohash6_point1'],-1)>=0 and index.get(adj_data.loc[i,'geohash6_point2'],-1)>=0:
            adj_mat[get_Timeindex(adj_data.loc[i,'date_id']),index[adj_data.loc[i,'geohash6_point1']],index[adj_data.loc[i,'geohash6_point2']],0]=int(adj_data.loc[i,'F_1'])
            adj_mat[get_Timeindex(adj_data.loc[i,'date_id']),index[adj_data.loc[i,'geohash6_point1']],index[adj_data.loc[i,'geohash6_point2']],1]=int(adj_data.loc[i,'F_2'])
    
    np.save('./dataset/adj_mat.npy',adj_mat)


if "__main__" == __name__:
    save_Index()
    # print(get_Index())

    save_Adjmatrix()