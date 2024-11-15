import copy
import json
import multiprocessing

import networkx
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch import LongTensor
from torch.utils.data import Dataset, DataLoader

from util.DataUtil import DataUtil


class TopologyAwareDataSplite:

    # 用于读取UCI数据集的初始化方法,div是时间转化换默认s转h
    def __init__(self, file_name=None, lstm_window=10, device="cpu",
                 node_feature=1, edge_feature=1, label_class=2, adj_file=None, dtw_file=None,
                 node_num=None, edge_num=None,
                 feature_file=None):

        if adj_file is None:
            return
        if adj_file.split('.')[1] != 'json':
            self.adj_matrix = np.load(adj_file)
            self.G = nx.from_numpy_array(self.adj_matrix)
            self.G = self.G.to_undirected()
        else:
            with open(adj_file, 'r') as json_file:
                self.G = json.load(json_file)

        if dtw_file is not None and dtw_file.split('.')[1] == 'json':

            with open(dtw_file, 'r') as d_file:
                self.dtw_sequence_dict = json.load(d_file)

        self.node_num = node_num
        self.edge_num = edge_num

        df = pd.read_excel(file_name, engine="openpyxl")
        self.data = df.to_numpy()
        self.lstm_window = lstm_window
        self.node_feature = node_feature
        self.label_class = label_class
        self.edge_feature = edge_feature
        self.device = device

        # 获取feature文件的数据
        # df = pd.read_excel(feature_file, engine="openpyxl")
        # feature_array = df.to_numpy()

        all_feature_tensor = torch.zeros(self.node_num + self.edge_num, node_feature)
        self.node_feature_tensor = all_feature_tensor[0:self.node_num]
        # self.edge_feature_tensor = all_feature_tensor[self.node_num: self.node_num + self.edge_num]
        # self.edge_index_dict = {feature_array[index, 2]: index - self.node_num for index in
        #                         range(self.node_num, feature_array[:, 0].size)}

        # 切分数据集作为测试数据集、验证数据集、和测试数据集
        self.data_length = len(self.data)
        dim = np.int32(self.node_num * node_feature)
        self.data = self.normalize_array_by_column(self.data, range(0, dim))
        label = self.data[:, 1].astype(np.float32)
        self.all_label = label
        self.data = self.data[:, 1:self.data.shape[1]]

        # 初始化节点的特征张量
        self.feature = torch.zeros((self.node_num, self.node_feature), dtype=torch.float32)

        self.feature[:, 1: self.feature.size(-1)] = self.node_feature_tensor[:, 1: self.feature.size(-1)]

        # 从数据集中读取故障序列信息，如果没有就设置为None
        self.sequence_info = None
        self.node_extra_feature = 0

        if self.data.shape[1] > dim:
            self.sequence_info = self.data[:, dim:self.data.shape[1]].astype(str)
            self.node_extra_feature = self.sequence_info.shape[1]

    def get_max_node(self):
        return np.max(self.data[:, 0])

    def normalize_array_by_column(self, data, indexes):
        for index in indexes:
            if len(data.shape) <= 1:
                column = data
            else:
                column = data[:, index]
            column[column == np.inf] = 1
            min_val = np.min(column)
            max_val = np.max(column)
            range_vals = max_val - min_val
            if range_vals == 0:
                range_vals = 1

            normal_col = (column - min_val) / range_vals
            if len(data.shape) <= 1:
                return normal_col
            else:
                data[:, index] = normal_col
        return data

    def build_sparse_matrix(self, adj_list: list, test_mode=False):
        adj_sparse_list = list()
        for adj in adj_list:
            if not test_mode:
                idx = adj.get("idx")[0]
                val = adj.get("value")[0]
            else:
                idx = adj.get("idx")
                val = adj.get("value")
            adj_sparse = torch.sparse_coo_tensor(idx, val, self.size, dtype=torch.float32)
            adj_sparse_list.append(adj_sparse.to(self.device))
        return adj_sparse_list

    def get_all_data_sample(self, dataset, data_length):
        feature_sample = list()
        adj_sample = list()
        delete_list = list()
        dtw_sample = list()

        for i in range(0, data_length):
            feature = self.feature.clone()

            if isinstance(self.G, networkx.Graph):
                tmp_coo = nx.to_scipy_sparse_array(self.G, format='coo')

                values = tmp_coo.data
                indices = np.vstack((tmp_coo.row, tmp_coo.col))
                index = torch.LongTensor(indices)
                v = torch.LongTensor(values)
                # # 这里先不转换为稀疏矩阵，不然dataloader会报错
                edge_index = {"idx": index, "value": v}
                adj_sample.append(edge_index)

            else:
                adj_tmp = self.G.get(str(i))
                if adj_tmp is None:
                    # self.all_label = np.delete(self.all_label, i, axis=0)
                    delete_list.append(i)
                    if self.dtw_sequence_dict is not None:
                        self.dtw_sequence_dict.pop(i, None)
                    continue
                adj_sample.append(adj_tmp)
                dtw_sample.append(self.dtw_sequence_dict.get(str(i)))

            for node_index in range(self.node_num):
                for feature_index in range(self.node_feature):
                    feature[node_index, feature_index] = dataset[i, np.int32(node_index * self.node_feature
                                                                             + feature_index)]

            feature_sample.append(feature)

        self.all_label = np.delete(self.all_label, delete_list, axis=0)

        return feature_sample, adj_sample, dtw_sample
