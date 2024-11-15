import base64
import pickle
import traceback

import numpy as np
import pandas as pd
import torch
from fastdtw import fastdtw
from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from extral.MAD import MAD


class DataUtil:

    @staticmethod
    def get_list_from_string(content: str):
        com_list = list()
        tmp = content.replace('[(', '').replace(')]', '').replace('), (', '-')
        for item in tmp.split('-'):

            item = list(filter(None, item.split(',')))  # 忽略空字符串
            try:
                item_tuple = tuple(map(np.int32, item))
            except ValueError:
                item_tuple = tuple(map(str, item[0].replace("\'", '')))
            com_list.append(item_tuple)
        return com_list

    @staticmethod
    def pre_deal_feature(fea):
        feature_dict = dict()
        for feature_data in fea:
            # feature_data = feature_data.reshape(-1, feature_data.size(-1))
            # 先转置，得到不同属性对应所有节点的特征值
            feature_data = feature_data.transpose(0, 1)

            num = feature_data.size(0)
            for i in range(num):
                target = feature_dict.get(i)
                if target is None:
                    feature_dict[i] = feature_data[i]
                else:
                    u_target = torch.cat([target, feature_data[i]], dim=-1)
                    feature_dict[i] = u_target
        feature = None
        for key in feature_dict.keys():  # 将list转换为Tensor表示
            tensor_item = feature_dict.get(key)
            feature_tensor = tensor_item.view(1, len(fea), -1)
            if feature is None:
                feature = feature_tensor
            else:
                feature = torch.cat([feature, feature_tensor], dim=0)

        return feature

    @staticmethod
    def code_nm_tensor_list_to_nparray(tensor_list: list):
        tmp = None
        for item in tensor_list:
            item = item.reshape(1, -1)
            if tmp is None:
                tmp = item
            else:
                tmp = torch.cat([tmp, item], dim=0)

        new_arr = tmp.numpy()

        return new_arr

    @staticmethod
    def decode_nparray_to_nm_tensor_list(arr: np.array, change_dim):
        tmp = list()
        for item in arr:
            item = torch.tensor(item)
            item = item.reshape(-1, change_dim)
            tmp.append(item)

        return tmp

    @staticmethod
    def save_cleaned_data(feature, label, file_name):
        writer = pd.ExcelWriter(file_name)
        arr = DataUtil.code_nm_tensor_list_to_nparray(feature)
        combined = np.concatenate((label, arr), axis=1)
        df = pd.DataFrame(combined)
        df.to_excel(writer, index=False)
        writer.close()

    @staticmethod
    def save_file_by_byte(adj_arr, file_name):
        writer = pd.ExcelWriter(file_name)
        df = pd.DataFrame(adj_arr)
        df.to_excel(writer, index=False)

    @staticmethod
    def decode_matrix_from_base64(bas64_str):
        binary_matrix = base64.b64decode(bas64_str)
        matrix = pickle.loads(binary_matrix)
        return matrix

    @staticmethod
    def encode_matrix_from_base64(adj_matrix):
        binary_matrix = pickle.dumps(adj_matrix)
        encode_stream = base64.b64encode(binary_matrix).decode('utf-8')
        return encode_stream

    @staticmethod
    def calculate_sequence_based_dtw(base_seq, sample_seq):
        try:
            distance, path = fastdtw(base_seq, sample_seq)
        except IndexError:
            distance = np.Inf
        return distance

    @staticmethod
    def calculate_sequence_based_mad(base_seq, sample_seq):
        mad = MAD(
            additional_cost=None,
            alpha=0.1,
            beta=0.01,
            first_step_DTW=False)
        try:
            mad.fit(X_source=base_seq.transpose(1, 2))
            Cost_OT, score_OT, all_iteration_time = mad.run_for_output(X_target=sample_seq.transpose(1, 2))
            distance = score_OT
            # print('OT=' + str(mad.OT))
            # print('Cost_OT=' + str(Cost_OT))
            print('score_OT=' + str(score_OT))
        except IndexError as e:
            print(e)
            traceback.print_exc()
            distance = np.Inf
        return distance

    @staticmethod
    def calculate_sequence_based_kshape(data, n_clusters):
        time_series = TimeSeriesScalerMeanVariance().fit_transform(data)
        ks = KShape(n_clusters=n_clusters, verbose=True).fit(time_series)
        dists = ks._cross_dists(time_series)
        labels = ks.predict(time_series)
        return dists, labels

    # 这个方法为了解决kshape没法识别较大的时间漂移，首先引入了ERP(Edit distance with Real Penalty)使的时间序列进行非线性对齐，
    # 解决时间漂移的问题之后在通过shape进行聚类
    @staticmethod
    def calculate_sequence_based_kshape_with_ERP(data, n_clusters):
        aligned_data = [DataUtil.align_series_with_erp(data[0], ts, gap_penalty=0.5) for ts in data]

        # 转换为适合 k-Shape 的格式
        aligned_data = np.array(aligned_data).reshape(len(aligned_data), -1, 1)

        time_series = TimeSeriesScalerMeanVariance().fit_transform(aligned_data)
        ks = KShape(n_clusters=n_clusters, verbose=True).fit(time_series)
        dists = ks._cross_dists(time_series)
        labels = ks.predict(time_series)
        return dists, labels

    @staticmethod
    def align_series_with_erp(reference_series, target_series, gap_penalty=0.5):
        """实现使用 ERP 对齐目标序列到参考序列的功能"""
        n = len(reference_series)
        m = len(target_series)

        # 初始化累积距离矩阵
        dp = np.zeros((n + 1, m + 1))
        dp[:, 0] = np.arange(n + 1) * gap_penalty
        dp[0, :] = np.arange(m + 1) * gap_penalty

        # 填充累积距离矩阵
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(reference_series[i - 1] - target_series[j - 1])
                dp[i, j] = min(dp[i - 1, j] + gap_penalty,  # 删除
                               dp[i, j - 1] + gap_penalty,  # 插入
                               dp[i - 1, j - 1] + cost)  # 匹配

        # 回溯路径进行对齐
        aligned_series = np.zeros_like(reference_series)
        i, j = n, m
        while i > 0 and j > 0:
            if dp[i, j] == dp[i - 1, j - 1] + abs(reference_series[i - 1] - target_series[j - 1]):
                aligned_series[i - 1] = target_series[j - 1]
                i -= 1
                j -= 1
            elif dp[i, j] == dp[i - 1, j] + gap_penalty:
                i -= 1
            else:
                j -= 1

        return aligned_series

    @staticmethod
    def normalize_array_by_column(data, indexes):  # 用于多inf的array的归一化
        for index in indexes:
            if len(data.shape) <= 1:
                column = data
            else:
                column = data[:, index]
            column[column == np.inf] = 10
            min_val = np.min(column)
            max_val = np.max(column)
            column[column == np.NAN] = max_val
            range_vals = max_val - min_val
            if range_vals == 0:
                range_vals = 1

            normal_col = (column - min_val) / range_vals
            if len(data.shape) <= 1:
                return normal_col
            else:
                data[:, index] = normal_col
        return data

    @staticmethod
    def normalize_array(data):  # 用于多inf的array的归一化
        data[data == -np.inf] = np.nan
        data[data == np.inf] = np.nan
        min_val = np.nanmin(data)
        max_val = np.nanmax(data)
        range_vals = max_val - min_val
        if range_vals == 0:
            range_vals = 1

        normal = (data - min_val) / range_vals
        return normal

    @staticmethod
    def normalize_tensor(data):  # 用于多inf的array的归一化
        min_val = data.min()
        max_val = data.max()
        range_vals = max_val - min_val
        if range_vals == 0:
            range_vals = 1

        normal = (data - min_val) / range_vals
        return normal

    @staticmethod
    # 该方法用于寻找adj中最小的前N个值，并替换为二元矩阵
    def get_bestN_index_in_matrix(adj_matrix, N):
        # 1. 将矩阵展平为一维数组
        flat_matrix = adj_matrix.ravel()

        # 2. 忽略 NaN 值，将 NaN 替换为负无穷大，以确保 NaN 不影响排序
        flat_matrix_no_nan = np.nan_to_num(flat_matrix, nan=np.inf)

        # 3. 找到前N个最大权重的索引
        top_N_indices = np.argsort(flat_matrix_no_nan)[:N]  # 从小到大排序

        # 4. 将展平的索引还原为二维矩阵的行列索引
        row_indices, col_indices = np.unravel_index(top_N_indices, adj_matrix.shape)

        # 5. 创建一个全为0的矩阵，准备替换
        binary_matrix = np.zeros_like(adj_matrix)

        # 6. 根据前 N 个最大值的索引，将对应位置替换为 1
        for row, col in zip(row_indices, col_indices):
            binary_matrix[row, col] = 1

        return binary_matrix
