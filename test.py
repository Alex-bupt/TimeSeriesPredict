import os
import pandas as pd
import CNN
import numpy as np
import torch

from data_process import TimeSeriesDataset, test_data_process

if "__main__" == __name__:
    node_data, edge_data = test_data_process()

    input_size = 35
    output_size = 2
    num_channels = [68] * 4
    kernel_size = 2
    dropout = 0.2

    model = CNN.TCN(input_size, output_size, num_channels, kernel_size, dropout)

    model_path = os.path.join("save", "best_model_25.pth")

    # 加载保存的模型参数
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()  # 设置模型为评估模式
        print("已加载保存的模型参数")
    else:
        print("无法找到保存的模型参数文件")

    test_data = np.array(node_data)

    X = test_data[:, 2:].astype(float)

    # 转换为PyTorch张量
    X = torch.tensor(X, dtype=torch.float32)

    y_result = []
    result = []

    # 将每行X放入测试
    with torch.no_grad():
        for i in range(X.shape[0]):
            y = model(X[i].view(1, 35, 1))
            y = y.view(2).numpy()
            y_result.append(y)

    # 读取 "node_data" 的第一列
    node_data_col1 = node_data["geohash_id"]

    # 读取 "y_result" 的第一列和第二列
    y_result_col1 = np.array(y_result)[:, 0]
    y_result_col2 = np.array(y_result)[:, 1]

    # 读取 "node_data" 的第二列
    node_data_col2 = node_data["date_id"]

    # 创建一个包含结果数据的DataFrame
    result_data = pd.DataFrame({
        "geohash_id": node_data_col1,
        "consumption_level": y_result_col1,
        "activity_level": y_result_col2,
        "date_id": node_data_col2
    })

    # 将结果数据按照所提供的格式合并为一列
    result_data["merged_column"] = result_data["geohash_id"] + "\t" + result_data["consumption_level"].astype(
        str) + "\t" + result_data["activity_level"].astype(str) + "\t" + result_data["date_id"].astype(str)

    # 保存结果数据到 "submit.csv" 文件，只保留一列
    result_data[["merged_column"]].to_csv("submit.csv", index=False,
                                          header=["geohash_id\tconsumption_level\tactivity_level\tdate_id"])