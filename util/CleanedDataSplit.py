import torch
from torch.utils.data import DataLoader, Dataset

from util.DataSplit import DataSplit


class CleanedDataSplit(DataSplit):

    def __init__(self, lstm_window=10, train=0.7, valid=0.2, device="cpu", node_feature=1, edge_feature=1,
                 label_class=1, node_num=None, edge_num=None, feature=None, adjs=None, label=None):

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
        end = int(data_length * train)

        # 这里是object对象，需要修改为float32
        label = torch.FloatTensor(label)
        self.label = label[start:end, 0]
        train_data = [feature[i] for i in range(start, end)]
        train_adjs = [adjs[i] for i in range(start, end)]
        # 测试数据集
        start = end + 1
        end = start + int(data_length * valid)
        test_data = [feature[i] for i in range(start, end)]
        test_adjs = [adjs[i] for i in range(start, end)]
        self.test_label = label[start:end, 0]
        # 验证数据集
        valid_data = [feature[i] for i in range(end + 1, data_length)]
        self.valid_label = label[end + 1:data_length, 0]
        valid_adj = [adjs[i] for i in range(end + 1, data_length)]

        self.adjs = adjs

        train_data = CleanedDatasetSplit(train_adjs, self.label, train_data, self.lstm_window)
        test_data = CleanedDatasetSplit(test_adjs, self.test_label, test_data, self.lstm_window)
        valid_data = CleanedDatasetSplit(valid_adj, self.valid_label, valid_data, self.lstm_window)

        self.train_dataLoader = DataLoader(train_data, batch_size=1, num_workers=0, shuffle=False)
        self.test_dataLoader = DataLoader(test_data, batch_size=1, num_workers=0, shuffle=False)
        self.valid_dataLoader = DataLoader(valid_data, batch_size=1, num_workers=0, shuffle=False)

        self.start = start
        self.end = end
        self.node_num = node_num

        self.size = (int(node_num), int(node_num))  # Note:这个要根据实际情况实际调整，大小要和稀疏矩阵和特征对应上

    def reset(self):
        pass


class CleanedDatasetSplit(Dataset):

    def __init__(self, adjs, label, feature, lstm_window):
        self.feature = feature
        self.lstm_window = lstm_window
        self.label = label
        self.adjs = adjs

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

        for i in range(idx - self.lstm_window, idx):
            mask.append(i)
            adj_sample.append(self.adjs[i])
            feature_sample.append(self.feature[i])

        return {"feature": feature_sample,
                "adj": adj_sample,
                "mask": mask,
                "feature_idx": [],
                "label": self.label}
