import concurrent.futures
import copy
import json

import networkx as nx
import numpy as np
import scipy as sp
import torch
from scipy.sparse import coo_matrix
from torch.utils.data import DataLoader, Dataset

from util.DataSplit import DataSplit
from util.DataUtil import DataUtil
from util.ExcelUtil import ExcelUtil


class CleanedDataSplit(DataSplit):

    def __init__(self, lstm_window=10, train=0.7, valid=0.2, device="cpu", node_feature=1, edge_feature=1,
                 label_class=1, node_num=None, terminal_node_num=None, feature=None, adjs=None, label=None,
                 extra_info=None,
                 dtws=None):

        # super().__init__(lstm_window, train, valid, device, node_feature, edge_feature, label_class, node_num, edge_num)
        self.lstm_window = lstm_window
        if feature is None and adjs is None and label is None:
            return

        data_length = len(feature)
        start = 0
        self.lstm_window = lstm_window
        self.node_feature = node_feature
        self.label_class = label_class
        self.device = device
        self.node_num = node_num
        self.terminal_node_num = terminal_node_num
        self.p = 0.4
        end = int(data_length * train)

        # 这里是object对象，需要修改为float32
        label = torch.FloatTensor(label)
        self.label = label[start:end, 0]
        train_data = [feature[i] for i in range(start, end)]
        train_adjs = [adjs[i] for i in range(start, end)]
        train_extra = None
        self.extra_node_feature = 0
        if extra_info is not None:  # extra_info 在前面的流程不处理，一般为np.array类型
            train_extra = extra_info[start:end, :]
            self.extra_node_feature = extra_info.shape[1]

        train_dtws = None
        train_dtws_matrixs = None
        if dtws is not None:
            # 初始化dtw序列，计算不同数据样本构件序列的dtw值，并且归一化
            dtws_matrixs = self.get_matrix_list_with_thread_pool(dtws, need_saved=False)
            train_dtws = [dtws[i] for i in range(start, end)]
            train_dtws_matrixs = [dtws_matrixs[i] for i in range(start, end)]

        # 测试数据集
        start = end + 1
        end = start + int(data_length * valid)
        test_data = [feature[i] for i in range(start, end)]
        test_adjs = [adjs[i] for i in range(start, end)]
        self.test_label = label[start:end, 0]
        test_extra = None
        if extra_info is not None:  # extra_info 在前面的流程不处理，一般为np.array类型
            test_extra = extra_info[start:end, :]

        test_dtws = None
        test_dtws_matrixs = None
        if dtws is not None:
            test_dtws = [dtws[i] for i in range(start, end)]
            test_dtws_matrixs = [dtws_matrixs[i] for i in range(start, end)]

        # 验证数据集
        valid_data = [feature[i] for i in range(end + 1, data_length)]
        self.valid_label = label[end + 1:data_length, 0]
        valid_adj = [adjs[i] for i in range(end + 1, data_length)]
        valid_extra = None
        if extra_info is not None:  # extra_info 在前面的流程不处理，一般为np.array类型
            valid_extra = extra_info[end + 1: data_length, :]

        valid_dtws = None
        valid_dtws_matrixs = None
        if dtws is not None:
            valid_dtws = [dtws[i] for i in range(end + 1, data_length)]
            valid_dtws_matrixs = [dtws_matrixs[i] for i in range(end + 1, data_length)]

        self.adjs = adjs
        self.size = (int(node_num), int(node_num))  # Note:这个要根据实际情况实际调整，大小要和稀疏矩阵和特征对应上

        train_data = CleanedDatasetSplit(train_adjs, self.label, train_data, self.lstm_window, train_extra, self.size,
                                         train_dtws, train_dtws_matrixs, terminal_node_num)
        test_data = CleanedDatasetSplit(test_adjs, self.test_label, test_data, self.lstm_window, test_extra, self.size,
                                        test_dtws, test_dtws_matrixs, terminal_node_num)
        valid_data = CleanedDatasetSplit(valid_adj, self.valid_label, valid_data, self.lstm_window, valid_extra
                                         , self.size, valid_dtws, valid_dtws_matrixs, terminal_node_num)

        self.train_dataLoader = DataLoader(train_data, batch_size=1, num_workers=0, shuffle=False)
        self.test_dataLoader = DataLoader(test_data, batch_size=1, num_workers=0, shuffle=False)
        self.valid_dataLoader = DataLoader(valid_data, batch_size=1, num_workers=0, shuffle=False)

        self.start = start
        self.end = end

    def reset(self):
        pass

    def get_dtws_feature_matrix(self, dtw):
        dtw_adj = np.zeros((self.node_num, self.node_num), dtype=np.float32)
        for xid in range(self.node_num):
            for yid in range(xid + 1):
                if xid >= self.node_num - self.terminal_node_num or xid == yid:
                    distance = np.nan
                else:
                    x_seq = dtw.get(str(xid))
                    y_seq = dtw.get(str(yid))
                    distance = DataUtil.calculate_sequence_based_dtw(x_seq, y_seq)
                dtw_adj[xid][yid] = distance
                dtw_adj[yid][xid] = distance

        dtw_adj = DataUtil.normalize_array(dtw_adj)
        dtw_adj = DataUtil.get_bestN_index_in_matrix(dtw_adj, int(self.node_num * self.node_num * self.p))
        print('算好啦!!!')

        return dtw_adj

    def get_matrix_list_with_thread_pool(self, dtws, need_saved=False):
        results = None
        try:
            with open('NEMData/network_1/tmp_data/tmp_dtws_adj.json') as tmp_file:
                json_data = json.load(tmp_file)
                tmp_datas = json_data.get("tmp")
                results = list()
                for tmp in tmp_datas:
                    dtw_adj = DataUtil.decode_matrix_from_base64(tmp)
                    results.append(dtw_adj)
                return results
        except FileNotFoundError:
            print("没有文件缓存，需要设置need_saved=true重新缓存文件")

            with concurrent.futures.ProcessPoolExecutor(max_workers=15) as executor:
                results = list(executor.map(self.get_dtws_feature_matrix, dtws))
                if need_saved:
                    tmp_datas = list()
                    for result in results:
                        data = DataUtil.encode_matrix_from_base64(result)
                        tmp_datas.append(data)
                    json_data = {"tmp": tmp_datas}
                    ExcelUtil.write_to_Json("NEMData/network_1/tmp_data/tmp_dtws_adj.json", json_data)

            return results


class CleanedDatasetSplit(Dataset):

    def __init__(self, adjs, label, feature, lstm_window, extra=None, adj_size=None, dtws=None, dtw_matrixs=None,
                 terminal_node_num=0):
        self.feature = feature
        self.lstm_window = lstm_window
        self.label = label
        self.adjs = adjs
        self.extra = extra
        self.adj_size = adj_size
        self.dtws = dtws
        self.dtw_matrixs = dtw_matrixs
        self.terminal_node_num = terminal_node_num

    def __len__(self):
        return len(self.feature) - self.lstm_window

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
        mask = list()
        extra_feature = list()
        sub_feature_sequence = list()

        for i in range(idx - self.lstm_window, idx):
            mask.append(i)
            feature = self.feature[i]
            # if self.dtws is not None:
            #     dtw_data = torch.FloatTensor(self.dtws[i,:])
            #     dtw_data = dtw_data.reshape(-1, 1)
            #     feature = torch.cat((feature, dtw_data), dim=-1)
            if self.extra is None:
                feature_sample.append(feature)
            else:
                sub_feature_sequence.clear()
                # 将字符串流base64解码为二进制，并转换为邻接矩阵，得到当前的拓扑
                cur_graph = None
                cur_adj_list = None
                if not isinstance(self.adjs[i], list):
                    adj_sparse = self.change_sparse_matrix(self.adjs[i])
                    cur_graph = nx.from_scipy_sparse_array(adj_sparse)
                    adj_sample.append(self.adjs[i])
                else:
                    cur_adj_list = copy.deepcopy(self.adjs[i])
                    cur_graph = self.get_current_graph_from_list(cur_adj_list)

                com_sequence = DataUtil.get_list_from_string(self.extra[i, 0])
                node_num = cur_graph.number_of_nodes()
                sparse_list = list()

                for coms_item in com_sequence:

                    tmp_graph = copy.deepcopy(cur_graph)

                    cur_extra_feature = torch.zeros(node_num, 1)
                    for com_id in coms_item:
                        if com_id != 'E':
                            tmp_graph.remove_node(com_id)
                            tmp_graph.add_node(com_id)

                        else:  # 说明出发了事件，需要使用当前事件对应的拓扑结构
                            if cur_adj_list is None:
                                print("存在错误，cur_adj_list 为 None")
                                return -1

                            cur_graph = self.get_current_graph_from_list(cur_adj_list)
                            tmp_graph = copy.deepcopy(cur_graph)

                        degrees = np.array([d for n, d in cur_graph.degree()])
                        max_degree = degrees.max()
                        min_degree = 0
                        val_range = max_degree - min_degree

                        # 计算当前拓扑所有度的变化情况
                        for cid in range(0, node_num):
                            try:
                                cur_degree = tmp_graph.degree(cid)
                                cur_degree = (cur_degree - min_degree) / val_range
                            except nx.exception.NetworkXError:
                                cur_degree = 0

                            cur_extra_feature[cid] = cur_degree

                        # feature = torch.cat((self.feature[i], cur_extra_feature), dim=-1)

                    sub_feature_sequence.append(cur_extra_feature)

                    if isinstance(self.adjs[i], list):
                        # 如果是考虑事件拓扑变化的情况，则需要在这里将图变化为稀疏矩阵
                        # tmp_coo = nx.to_scipy_sparse_array(tmp_graph, format='coo')
                        origin_adj = nx.to_numpy_array(tmp_graph)
                        temporal_adj = np.multiply(origin_adj, self.dtw_matrixs[i])
                        tmp_coo = coo_matrix(temporal_adj)

                        values = tmp_coo.data
                        indices = np.vstack((tmp_coo.row, tmp_coo.col))
                        index = torch.LongTensor(indices)
                        v = torch.LongTensor(values)
                        # # 这里先不转换为稀疏矩阵，不然dataloader会报错
                        edge_index = {"idx": index, "value": v}
                        sparse_list.append(edge_index)

                    del tmp_graph

                if isinstance(self.adjs[i], list):
                    adj_sample.append(copy.deepcopy(sparse_list))

                sparse_list.clear()

                del cur_adj_list

                # if isinstance(self.adjs[i], list):
                #     # 如果是考虑事件拓扑变化的情况，则需要在这里将图变化为稀疏矩阵
                #     tmp_coo = nx.to_scipy_sparse_array(cur_graph, format='coo')
                #
                #     values = tmp_coo.data
                #     indices = np.vstack((tmp_coo.row, tmp_coo.col))
                #     index = torch.LongTensor(indices)
                #     v = torch.LongTensor(values)
                #     # # 这里先不转换为稀疏矩阵，不然dataloader会报错
                #     edge_index = {"idx": index, "value": v}
                #     adj_sample.append(edge_index)

                feature_sample.append(feature)
                # feature_sample.append(copy.deepcopy(sub_feature_sequence))
                performance_feature_sequence = self.get_performance_feature_tensor_base_index(i, node_num,
                                                                                              len(sub_feature_sequence))

                combine_feature = list()
                for sub_feature, performance_feature in zip(sub_feature_sequence, performance_feature_sequence):
                    combine = torch.cat((sub_feature, performance_feature), dim=-1)
                    combine_feature.append(combine)

                extra_feature.append(copy.deepcopy(combine_feature))
                sub_feature_sequence.clear()
                del performance_feature_sequence
                del combine_feature

        return {"feature": feature_sample,
                "adj": adj_sample,
                "mask": mask,
                "feature_idx": [],
                "label": self.label,
                "extra_feature": extra_feature}

    def change_sparse_matrix(self, adj):
        idx = adj.get("idx")
        val = adj.get("value")
        adj_sparse = sp.sparse.coo_matrix((val, idx), self.adj_size)
        return adj_sparse

    def get_current_graph_from_list(self, current_adjs):
        base64_matrix = current_adjs[0]
        current_adjs.pop(0)
        matrix = DataUtil.decode_matrix_from_base64(base64_matrix)
        cur_graph = nx.from_numpy_array(matrix)

        return cur_graph

    def get_performance_feature_tensor_base_index(self, index, node_num, feature_length):
        feature_dict = self.dtws[index]
        performance_extra_feature = list()
        for i in range(feature_length):
            sub_performance_extra_feature = torch.zeros(node_num, 1)
            for key in feature_dict.keys():
                node_performance_sequence = feature_dict.get(key)
                if int(key) >= node_num - self.terminal_node_num:
                    continue

                if len(node_performance_sequence) != feature_length:
                    print("数据匹配错误~")
                    return None
                sub_performance_extra_feature[int(key)] = node_performance_sequence[i]
            performance_extra_feature.append(DataUtil.normalize_tensor(sub_performance_extra_feature))

        return performance_extra_feature
