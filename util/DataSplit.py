import copy
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


class DataSplit:

    # 用于读取UCI数据集的初始化方法,div是时间转化换默认s转h
    def __init__(self, file_name=None, skip_rows=0, div=3600, lstm_window=10, train=0.7, valid=0.2, device="cpu",
                 node_feature=1, edge_feature=1, label_class=2, adj_file=None, node_num=None, edge_num=None,
                 feature_file=None):

        # if mode == "UCI":
        #     self.data = np.loadtxt(fname=file_name, skiprows=skip_rows)
        #     self.time = self.data[:, 3]
        #     self.trans_time = (self.time - self.time[0]) / div
        #     self.data[:, 3] = self.trans_time
        #     # self.data[:, [0, 1]] = self.data[:, [0, 1]] - starting
        #     self.lstm_window = lstm_window
        #     self.node_feature = node_feature
        #     self.label_class = label_class
        #     self.device = device
        #     self.edge_feature = self.node_feature * 2
        #
        #     # 切分数据集作为测试数据集、验证数据集、和测试数据集
        #     start = 0
        #     data_length = len(self.data)
        #     end = int(data_length * train)
        #     train_data = self.data[start:end]
        #     start = end + 1
        #     end = start + int(data_length * valid)
        #     self.test_data = self.data[start:end]
        #     self.G = nx.Graph()
        #     self.test_G = None
        #     self.node_num = int(self.get_node_num(0))
        #     self.G.add_nodes_from(list(range(1, self.node_num)))
        #     self.label = torch.zeros(int((self.node_num * self.node_num)), 2, dtype=torch.float32).to(self.device)
        #     self.label[:, 0] = 1
        #     train_data = datasetSplit(self.G, self.label, train_data, self.lstm_window, self.node_num,
        #                               self.node_feature)
        #     # test_data = datasetSplit(self.G, self.label, test_data, self.lstm_window, self.node_num, self.node_feature)
        #
        #     self.train_dataLoader = DataLoader(train_data, batch_size=1, num_workers=0, shuffle=False)
        #     # self.test_dataLoader = DataLoader(test_data, batch_size=1, num_workers=0, shuffle=False)
        #     self.valid_data = self.data[end + 1:data_length]
        #     self.train_dataLoader = DataLoader(train_data, batch_size=1, num_workers=0, shuffle=False)
        #     self.start = start
        #     self.end = end
        #     self.size = (int(self.node_num), int(self.node_num))  # Note:这个要根据实际情况实际调整，大小要和稀疏矩阵和特征对应上
        # else:

        if adj_file is None:
            return
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
        df = pd.read_excel(feature_file, engine="openpyxl")
        feature_array = df.to_numpy()

        # fea_tmp = feature_array[:, 0:2].astype(np.float32)
        # all_feature_tensor = torch.tensor(fea_tmp)
        # all_feature_tensor = self.normalize_columns(all_feature_tensor)

        all_feature_tensor = torch.zeros(self.node_num + self.edge_num, node_feature)
        self.node_feature_tensor = all_feature_tensor[0:self.node_num]
        self.edge_feature_tensor = all_feature_tensor[self.node_num: self.node_num + self.edge_num]
        self.edge_index_dict = {feature_array[index, 2]: index - self.node_num for index in
                                range(self.node_num, feature_array[:, 0].size)}

        # 切分数据集作为测试数据集、验证数据集、和测试数据集
        start = 0
        data_length = len(self.data)
        self.data = self.normalize_array_by_column(self.data, [1, 5, 6])
        end = int(data_length * train)

        # 这里是object对象，需要修改为float32
        label = self.data[:, 1].astype(np.float32)
        self.all_label = label
        label = torch.FloatTensor(label).to(self.device)
        label = label.unsqueeze(1)
        # 训练数据集
        self.label = label[start:end, 0]
        train_data = self.data[start:end]
        # 测试数据集
        start = end + 1
        end = start + int(data_length * valid)
        test_data = self.data[start:end]
        self.test_label = label[start:end, 0]
        # 验证数据集
        valid_data = self.data[end + 1:data_length]
        self.valid_label = label[end + 1:data_length, 0]

        self.adj_matrix = np.load(adj_file)
        self.G = nx.from_numpy_array(self.adj_matrix)
        self.G = self.G.to_undirected()

        # 初始化节点的特征张量
        self.feature = torch.zeros((self.node_num, self.node_feature), dtype=torch.float32)
        # degrees = dict(self.G.degree())

        # 初始化节点的度特征
        degrees = np.array([d for n, d in self.G.degree()])
        self.max_degree = degrees.max()
        self.min_degree = 0
        normalized_degrees = (degrees - self.min_degree) / (self.max_degree - self.min_degree)
        for index in range(normalized_degrees.size):
            self.feature[index, 0] = normalized_degrees[index]

        self.feature[:, 1: self.feature.size(-1)] = self.node_feature_tensor[:, 1: self.feature.size(-1)]

        self.backup_feature = copy.deepcopy(self.feature)

        # train_data = NEMdatasetSplit(self.G, self.label, train_data, self.lstm_window, self.max_degree,
        #                              self.min_degree,
        #                              self.feature)
        # test_data = NEMdatasetSplit(self.G, self.test_label, test_data, self.lstm_window, self.max_degree,
        #                             self.min_degree,
        #                             self.feature)
        # valid_data = NEMdatasetSplit(self.G, self.valid_label, valid_data, self.lstm_window, self.max_degree,
        #                              self.min_degree,
        #                              self.feature)
        #
        # self.train_dataLoader = DataLoader(train_data, batch_size=1, num_workers=0, shuffle=False)
        # self.test_dataLoader = DataLoader(test_data, batch_size=1, num_workers=0, shuffle=False)
        # self.valid_dataLoader = DataLoader(valid_data, batch_size=1, num_workers=0, shuffle=False)

        self.start = start
        self.end = end
        self.size = (int(self.node_num), int(self.node_num))  # Note:这个要根据实际情况实际调整，大小要和稀疏矩阵和特征对应上

    def clear(self):
        self.G.clear_edges()
        self.label = torch.zeros(int((self.node_num * self.node_num)), 2, dtype=torch.float32).to(self.device)
        self.label[:, 0] = 1

    def reset(self):
        self.G = nx.from_numpy_array(self.adj_matrix)
        self.feature = self.backup_feature

    def get_node_num(self, start):
        return self.get_max_node() - start

    def get_max_node(self):
        return np.max(self.data[:, 0])

    def normalize_array_by_column(self, data, indexes):
        for index in indexes:
            if len(data.shape) <= 1:
                column = data
            else:
                column = data[:, index]
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

    # tensor归一化
    def normalize_columns(self, tensor):
        # 将 inf 和 -inf 替换为 NaN
        tensor[tensor == float('inf')] = float('nan')
        tensor[tensor == float('-inf')] = float('nan')
        consider_nan = False

        # 创建掩码，标记非 NaN 值
        mask = ~torch.isnan(tensor)
        if mask.size(0) != tensor.size(0):
            consider_nan = True

        # 计算每一列的最小值和最大值，忽略 NaN
        min_vals = torch.where(mask, tensor, torch.tensor(float('inf'))).min(dim=0, keepdim=True).values
        max_vals = torch.where(mask, tensor, torch.tensor(float('-inf'))).max(dim=0, keepdim=True).values

        # 计算每一列的范围，处理除零情况
        range_vals = max_vals - min_vals
        if consider_nan:
            range_vals += range_vals * 0.1
        range_vals[range_vals == 0] = 1.0  # 你也可以选择其他非零值

        # 归一化处理
        normalized_tensor = (tensor - min_vals) / range_vals

        # 将 NaN 替换为 1.0 或其他合理值
        normalized_tensor[torch.isnan(normalized_tensor)] = 1.0

        return normalized_tensor

    def build_sparse_matrix(self, adj_list: list, test_mode=False):
        adj_sparse_list = list()
        for adj in adj_list:
            if isinstance(adj, list):
                adj_sparse = list()
                for seq_adj in adj:
                    idx = seq_adj.get("idx")[0]
                    val = seq_adj.get("value")[0]
                    seq_adj_sparse = torch.sparse_coo_tensor(idx, val, self.size, dtype=torch.float32).to(self.device)
                    adj_sparse.append(seq_adj_sparse)
            else:
                if not test_mode:
                    idx = adj.get("idx")[0]
                    val = adj.get("value")[0]
                else:
                    idx = adj.get("idx")
                    val = adj.get("value")
                adj_sparse = torch.sparse_coo_tensor(idx, val, self.size, dtype=torch.float32).to(self.device)
            adj_sparse_list.append(adj_sparse)
        return adj_sparse_list

    def get_test_data_feature(self):
        if self.test_G is None:
            self.test_G = nx.Graph()
            for i in range(0, self.start):
                self.test_G.add_edge(self.data[i, 0], self.data[i, 1])

        adj_list = list()
        feature_list = list()
        node_mask = list()
        feature = torch.zeros((self.node_num, self.node_feature), dtype=torch.float32)

        for i in range(0, self.end):
            self.label[int((self.data[i, 0]) * self.node_num + (self.data[i, 1]))] \
                = torch.IntTensor([0, 1])
            self.label[int((self.data[i, 1]) * self.node_num + (self.data[i, 0]))] \
                = torch.IntTensor([0, 1])

            if i < self.start:
                self.test_G.add_edge(self.data[i, 0], self.data[i, 1])

            try:
                s_degree = self.test_G.degree(self.data[i, 0])
            except networkx.exception.NetworkXError:
                s_degree = 0

            try:
                d_degree = self.test_G.degree(self.data[i, 1])
            except networkx.exception.NetworkXError:
                d_degree = 0

            feature_v = (((s_degree + d_degree) * self.data[i, 2]) / (self.node_num + self.node_num))
            sidx = int(self.data[i, 0])  # 源节点索引
            didx = int(self.data[i, 1])  # 目的节点的索引
            feature[sidx, 0] = feature_v
            feature[didx, 0] = feature_v

            if i >= self.start:
                node_mask.append((int(self.data[i, 0]), int(self.data[i, 1])))
            # adj_list.append(edge_index)
            # feature_list.append(feature.clone().to(self.device))

        tmp_coo = nx.to_scipy_sparse_array(self.test_G, format='coo')
        # tmp_coo = sp.coo_matrix(adj)
        values = tmp_coo.data
        indices = np.vstack((tmp_coo.row, tmp_coo.col))
        index = torch.LongTensor(indices)
        v = torch.LongTensor(values)
        # # 这里先不转换为稀疏矩阵，不然dataloader会报错
        edge_index = {"idx": index, "value": v}
        adj_list.append(edge_index)
        feature_list.append(feature.to(self.device))

        return adj_list, feature_list, node_mask

    def get_all_data_sample(self, dataset, data_length):
        feature_sample = list()
        adj_sample = list()

        for i in range(0, data_length):
            feature = self.feature.clone()

            edge_list = DataUtil.get_list_from_string(dataset[i, 2])
            # print(edge_list)

            option = str(dataset[i, 4])

            if option == NEMdatasetSplit.RMOVE:
                self.G.remove_edges_from(edge_list)
                # for edge in edge_list:
                #     self.G.remove_edge(edge[0], edge[1])

            # elif option == NEMdatasetSplit.ADD:
            #     # self.G.add_edges_from(edge_list)
            #     for edge in edge_list:
            #         self.G.add_edge(edge[0], edge[1], weight=edge[2])

            tmp_coo = nx.to_scipy_sparse_array(self.G, format='coo')

            values = tmp_coo.data
            indices = np.vstack((tmp_coo.row, tmp_coo.col))
            index = torch.LongTensor(indices)
            v = torch.LongTensor(values)
            # # 这里先不转换为稀疏矩阵，不然dataloader会报错
            edge_index = {"idx": index, "value": v}
            adj_sample.append(edge_index)

            for edg in edge_list:
                # if bool_dict.get(edg[0]) is None:
                s_degree = (self.G.degree(edg[0]) - self.min_degree) / (self.max_degree - self.min_degree)
                # bool_dict.setdefault(edg[0], True)
                feature[edg[0], 0] = s_degree
                # self.feature[edg[0], 1] = self.dataset[i, 5]
                # self.feature[edg[0], 2] = self.dataset[i, 6]

                # if bool_dict.get(edg[1]) is None:
                d_degree = (self.G.degree(edg[1]) - self.min_degree) / (self.max_degree - self.min_degree)
                # bool_dict.setdefault(edg[1], True)
                feature[edg[1], 0] = d_degree
                # self.feature[edg[1], 1] = self.dataset[i, 5]
                # self.feature[edg[1], 2] = self.dataset[i, 6]

            feature[np.int32(dataset[i, 3]), 0] = dataset[i, 5]
            feature[np.int32(dataset[i, 3]), 1] = dataset[i, 6]

            # 把边加回去
            for edge in edge_list:
                self.G.add_edge(edge[0], edge[1], weight=edge[2])

            feature_sample.append(feature)

        return feature_sample, adj_sample


class datasetSplit(Dataset):
    # lock = multiprocessing.Lock()

    def __init__(self, G: networkx.Graph, label, dataset, lstm_window, node_num, node_feature_dim):
        self.dataset = dataset
        self.lstm_window = lstm_window
        self.label = label
        self.G = G
        # self.multiproccess_graph = multiprocessing.Manager().dict()
        # self.multiproccess_graph.setdefault("G_adj", nx.to_numpy_array(self.G))
        self.adj_dict = dict()  # 用于保存已经转换过的稀疏矩阵，这样就不用通过nx重复加载浪费时间了
        self.node_num = node_num
        # self.node_degree = multiprocessing.Manager().dict()
        self.feature = torch.zeros((self.node_num, node_feature_dim), dtype=torch.float32)

    def __len__(self):
        return len(self.dataset) - self.lstm_window

    def __getitem__(self, idx):
        sample = self.get_samlpe(idx + self.lstm_window)
        return sample

    # def get_node_num(self):
    #     return len(np.unique(self.dataset))

    def get_samlpe(self, idx):
        feature_sample = list()
        adj_sample = list()
        node_mask = list()

        for i in range(idx - self.lstm_window, idx):
            target_adj = self.adj_dict.get(i)
            if target_adj is None:
                # G_adj = self.multiproccess_graph.get("G_adj")
                # if G_adj is None:
                #     return {}
                #
                # self.G = nx.from_numpy_array(G_adj)

                self.G.add_edge(self.dataset[i, 0], self.dataset[i, 1])
                self.label[int((self.dataset[i, 0]) * self.node_num + (self.dataset[i, 1]))] \
                    = torch.IntTensor([0, 1])
                self.label[int((self.dataset[i, 1]) * self.node_num + (self.dataset[i, 0]))] \
                    = torch.IntTensor([0, 1])
                # 提出节点出度和入度的特征
                # self.node_degree[self.dataset[i, 0]] = self.G.degree(self.dataset[i, 0])
                # self.node_degree[self.dataset[i, 1]] = self.G.degree(self.dataset[i, 1])

                # new_adj = nx.to_numpy_array(self.G)
                # self.multiproccess_graph["G_adj"] = new_adj

                tmp_coo = nx.to_scipy_sparse_array(self.G, format='coo')
                # tmp_coo = sp.coo_matrix(adj)
                values = tmp_coo.data
                indices = np.vstack((tmp_coo.row, tmp_coo.col))
                index = torch.LongTensor(indices)
                v = torch.LongTensor(values)
                # # 这里先不转换为稀疏矩阵，不然dataloader会报错
                # edge_index = torch.sparse_coo_tensor(i, v, tmp_coo.shape)
                edge_index = {"idx": index, "value": v}

                self.adj_dict.setdefault(i, edge_index)

            else:
                edge_index = target_adj

            node_mask.append((int(self.dataset[i, 0]), int(self.dataset[i, 1])))

            s_degree = self.G.degree(self.dataset[i, 0])
            d_degree = self.G.degree(self.dataset[i, 1])

            feature_v = (((s_degree + d_degree) * self.dataset[i, 2]) / (self.node_num + self.node_num))
            sidx = int(self.dataset[i, 0])  # 源节点索引
            didx = int(self.dataset[i, 1])  # 目的节点的索引
            self.feature[sidx, 0] = feature_v
            self.feature[didx, 0] = feature_v
            adj_sample.append(edge_index)
            feature_sample.append(self.feature.clone())

        return {"feature": feature_sample,
                "adj": adj_sample,
                "label": self.label,
                "node_mask": node_mask}


class NEMdatasetSplit(Dataset):
    # lock = multiprocessing.Lock()
    RMOVE = "remove"
    ADD = "add"

    def __init__(self, G: networkx.Graph, label, dataset, lstm_window, max_degree, min_degree, feature):
        self.dataset = dataset
        self.lstm_window = lstm_window
        self.label = label
        self.G = G
        # self.multiproccess_graph = multiprocessing.Manager().dict()
        # self.multiproccess_graph.setdefault("G_adj", nx.to_numpy_array(self.G))
        self.adj_dict = dict()  # 用于保存已经转换过的稀疏矩阵，这样就不用通过nx重复加载浪费时间了
        self.max_degree = max_degree
        self.min_degree = min_degree
        # self.node_degree = multiprocessing.Manager().dict()
        self.feature = feature

    def __len__(self):
        return len(self.dataset) - self.lstm_window

    def __getitem__(self, idx):
        sample = self.get_samlpe(idx + self.lstm_window)
        return sample

    # def get_list_from_string(self, content: str):
    #     edg_list = list()
    #     tmp = content.replace('[(', '').replace(')]', '').replace('), (', '-')
    #     for item in tmp.split('-'):
    #         item_tuple = tuple(map(np.int32, item.split(', ')))
    #         edg_list.append(item_tuple)
    #     return edg_list

    def get_samlpe(self, idx):
        feature_sample = list()
        adj_sample = list()
        feature_idx = list()
        mask = list()

        for i in range(idx - self.lstm_window, idx):
            mask.append(i)
            target_adj = self.adj_dict.get(i)
            # 获取需要变化的边
            edge_list = DataUtil.get_list_from_string(self.dataset[i, 2])
            # print(edge_list)

            if target_adj is None:
                # G_adj = self.multiproccess_graph.get("G_adj")
                # if G_adj is None:
                #     return {}
                #
                # self.G = nx.from_numpy_array(G_adj)
                option = str(self.dataset[i, 4])

                if option == NEMdatasetSplit.RMOVE:
                    self.G.remove_edges_from(edge_list)
                    # for edge in edge_list:
                    #     self.G.remove_edge(edge[0], edge[1])

                # elif option == NEMdatasetSplit.ADD:
                #     # self.G.add_edges_from(edge_list)
                #     for edge in edge_list:
                #         self.G.add_edge(edge[0], edge[1], weight=edge[2])

                tmp_coo = nx.to_scipy_sparse_array(self.G, format='coo')

                values = tmp_coo.data
                indices = np.vstack((tmp_coo.row, tmp_coo.col))
                index = torch.LongTensor(indices)
                v = torch.LongTensor(values)
                # # 这里先不转换为稀疏矩阵，不然dataloader会报错
                # edge_index = torch.sparse_coo_tensor(i, v, tmp_coo.shape)
                edge_index = {"idx": index, "value": v}

                self.adj_dict.setdefault(i, edge_index)

            else:
                edge_index = target_adj

            # 重新计算各节点变化后的特征
            # bool_dict = dict()
            for edg in edge_list:
                # if bool_dict.get(edg[0]) is None:
                s_degree = (self.G.degree(edg[0]) - self.min_degree) / (self.max_degree - self.min_degree)
                # bool_dict.setdefault(edg[0], True)
                self.feature[edg[0], 0] = s_degree
                # self.feature[edg[0], 1] = self.dataset[i, 5]
                # self.feature[edg[0], 2] = self.dataset[i, 6]

                # if bool_dict.get(edg[1]) is None:
                d_degree = (self.G.degree(edg[1]) - self.min_degree) / (self.max_degree - self.min_degree)
                # bool_dict.setdefault(edg[1], True)
                self.feature[edg[1], 0] = d_degree
                # self.feature[edg[1], 1] = self.dataset[i, 5]
                # self.feature[edg[1], 2] = self.dataset[i, 6]
                feature_idx.append((edg[0], edg[1]))

            self.feature[np.int32(self.dataset[i, 3]), 1] = self.dataset[i, 5]
            self.feature[np.int32(self.dataset[i, 3]), 2] = self.dataset[i, 6]
            adj_sample.append(edge_index)
            feature_sample.append(self.feature.clone())

            # 把边加回去
            for edge in edge_list:
                self.G.add_edge(edge[0], edge[1], weight=edge[2])

        return {"feature": feature_sample,
                "adj": adj_sample,
                "mask": mask,
                "feature_idx": feature_idx,
                "label": self.label}
