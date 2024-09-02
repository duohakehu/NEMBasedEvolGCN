import base64
import pickle

import numpy as np
import pandas as pd
import torch


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
