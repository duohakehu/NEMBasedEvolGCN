import copy

import torch
from torch import nn
from torch.nn import LSTM, RReLU, Linear, AdaptiveAvgPool1d, ReLU
from torch_geometric.nn import GCNConv, MLP, GATConv

from Classifier import Predict
from util.DataUtil import DataUtil
from util.FeatureDict import FeatureDict


class EvolGCN(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim, num_class, num_layers, device, node_num):
        super(EvolGCN, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        # self.node_feature_weight = nn.Embedding(input_channel, node_em_dim)
        self.lstm = LSTM(input_size=node_num, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        # self.stack_lin = Linear(in_features=input_dim, out_features=hidden_dim)
        self.mlp = MLP(in_channels=hidden_dim, out_channels=node_num, hidden_channels=hidden_dim,
                       num_layers=num_layers)
        # self.weight = nn.Embedding(num_embeddings= input_data_size, embedding_dim=input_dim).weight.data
        self.gcn_conv = GCNConv(input_dim, output_dim)
        # self.gcn_conv.reset_parameters()  # 初始化权重
        # self.weight = self.gcn_conv.lin.weight.data.to(self.device)
        self.reLu = ReLU()
        self.cmlp = MLP(in_channels=input_dim, out_channels=output_dim, hidden_channels=hidden_dim,
                        num_layers=num_layers)

        self.feature_recent_dict = FeatureDict()
        self.feature_linear = nn.Linear(in_features=input_dim * 2, out_features=output_dim)
        self.lin = nn.Linear(input_dim, input_dim)
        # self.lin_p = nn.Linear(node_num, num_class)
        self.predict = Predict(input_dim * node_num, num_class)
        # 全局池化
        self.global_avg_pool = AdaptiveAvgPool1d(num_class)
        #
        # for m in self.modules():
        #     if isinstance(m, GCNConv):
        #         for name, param in self.named_parameters():
        #             if "gcn_conv" in name and "weight" in name:
        #                 nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, Linear):
        #         nn.init.constant_(m.weight, 0)
        #         nn.init.constant_(m.bias, 0)

    # def build_time_sequence_for_feature(self, feature_list, idx_list):
    #     prepapre_feature_dict = FeatureDict()
    #
    #     for f_t, feature_t in enumerate(feature_list):
    #         for id_t, idx in enumerate(idx_list):
    #             sidx = idx[0]
    #             didx = idx[1]
    #             is_sidx_linear = False
    #             is_didx_linear = False
    #             feature_t = feature_t.reshape(-1, feature_t.size(-1))
    #             feature_t = feature_t.to(self.device)
    #
    #             if self.feature_recent_dict.get(sidx) is None:
    #                 feature_t_for_sidx = feature_t[sidx]  # sidx的特征
    #             else:
    #                 feature_t_for_sidx = self.feature_recent_dict.get(sidx)
    #                 is_sidx_linear = True
    #
    #             if self.feature_recent_dict.get(didx) is None:
    #                 feature_t_for_didx = feature_t[didx]  # didx的特征
    #             else:
    #                 feature_t_for_didx = self.feature_recent_dict.get(didx)
    #                 is_didx_linear = True
    #
    #             if f_t == id_t:  # 说明当前时间发生了交互需要加入新的特征
    #                 if is_sidx_linear:
    #                     tmp = torch.cat((feature_t_for_sidx, feature_t_for_didx), dim=1)
    #                     feature_t_for_didx = self.feature_linear(tmp)
    #                 if is_didx_linear:
    #                     tmp = torch.cat((feature_t_for_sidx, feature_t_for_didx), dim=1)
    #                     feature_t_for_didx = self.feature_linear(tmp)
    #
    #             if prepapre_feature_dict.get(sidx) is None:
    #                 prepapre_feature_dict.setdefault(sidx, [feature_t_for_sidx])
    #             else:
    #                 sidx_list = prepapre_feature_dict.get(sidx)
    #                 feature_len = len(sidx_list)
    #                 if feature_len <= f_t:
    #                     sidx_list.append(feature_t_for_sidx)
    #                 else:
    #                     sidx_list[-1] = feature_t_for_sidx  # 之前已经做过特征交互了，放弃老的特征，更新为最新的特征
    #
    #             if prepapre_feature_dict.get(didx) is None:
    #                 prepapre_feature_dict.setdefault(didx, [feature_t_for_didx])
    #             else:
    #                 didx_list = prepapre_feature_dict.get(didx)
    #                 feature_len = len(didx_list)
    #                 if feature_len <= f_t:
    #                     didx_list.append(feature_t_for_didx)
    #                 else:
    #                     didx_list[-1] = feature_t_for_didx  # 之前已经做过特征交互了，放弃老的特征，更新为最新的特征
    #
    #     for key in prepapre_feature_dict.keys():  # 将list转换为Tensor表示
    #         tensor_list = prepapre_feature_dict.get(key)
    #         feature_tensor = torch.tensor(tensor_list)
    #         feature_tensor = feature_tensor.reshape(1, feature_tensor.size(-1), -1)
    #         prepapre_feature_dict[key] = feature_tensor.to(device=self.device)
    #
    #     return prepapre_feature_dict

    def build_time_sequence_for_feature(self, feature_list, idx_list):
        prepapre_feature_dict = FeatureDict()

        for f_t, feature_t in enumerate(feature_list):
            for id_t, idx in enumerate(idx_list):
                sidx = idx[0]
                didx = idx[1]

                feature_t = feature_t.reshape(-1, feature_t.size(-1))
                feature_t = feature_t.to(self.device)

                feature_t_for_sidx = feature_t[sidx]  # sidx的特征

                feature_t_for_didx = feature_t[didx]  # didx的特征

                if prepapre_feature_dict.get(sidx) is None:
                    prepapre_feature_dict.setdefault(sidx, [feature_t_for_sidx])
                else:
                    sidx_list = prepapre_feature_dict.get(sidx)
                    feature_len = len(sidx_list)
                    if feature_len <= f_t:
                        sidx_list.append(feature_t_for_sidx)
                    else:
                        sidx_list[-1] = feature_t_for_sidx  # 之前已经做过特征交互了，放弃老的特征，更新为最新的特征

                if prepapre_feature_dict.get(didx) is None:
                    prepapre_feature_dict.setdefault(didx, [feature_t_for_didx])
                else:
                    didx_list = prepapre_feature_dict.get(didx)
                    feature_len = len(didx_list)
                    if feature_len <= f_t:
                        didx_list.append(feature_t_for_didx)
                    else:
                        didx_list[-1] = feature_t_for_didx  # 之前已经做过特征交互了，放弃老的特征，更新为最新的特征

        for key in prepapre_feature_dict.keys():  # 将list转换为Tensor表示
            tensor_list = prepapre_feature_dict.get(key)
            feature_tensor = torch.tensor(tensor_list)
            feature_tensor = feature_tensor.reshape(1, feature_tensor.size(-1), -1)
            prepapre_feature_dict[key] = feature_tensor.to(device=self.device)

        return prepapre_feature_dict

    ##链路预测的，感觉feature_data[idx_list[-1][0]] = data很奇怪
    # def forward(self, fea, idx_list, edge_list, test_mode=False):
    #     # 通过LSTM方法更新weight，而不是通过GCN来更新,LSTM输入注意input包含 [batchSize, timeSql, feature]
    #     # self.weight = self.gcn_conv.lin.weight.data.to(self.device)
    #     # weight = self.lstm(self.weight)[0]
    #     # w = self.mlp(weight)
    #     # x = self.lstm(x)[0]
    #     # x = self.mlp(x)
    #     # #更新weight后在进行图卷
    #     # self.update_weight(w)
    #     if not test_mode:
    #         prepapre_feature_dict = self.build_time_sequence_for_feature(fea, idx_list)
    #         feature_data = fea[-1]
    #         for idx_key in prepapre_feature_dict:
    #             feature = prepapre_feature_dict.get(idx_key)
    #             h_t, (hn, cn) = self.lstm(feature)
    #             hn = hn.reshape(-1, self.hidden_dim)
    #             cn = hn.reshape(-1, self.hidden_dim)
    #             data = torch.cat([hn, cn], dim=1)
    #             data = self.mlp(data)
    #             # 更新节点特征作为下一次使用
    #             # self.feature_recent_dict[idx_key] = data
    #             # 更新到下一步进行图卷积的特征矩阵中
    #             feature_data = feature_data.reshape(-1, feature_data.size(-1))
    #
    #             feature_data[idx_list[-1][0]] = data
    #             feature_data[idx_list[-1][1]] = data
    #
    #         feature_data = feature_data.to(self.device)
    #         edge_index = edge_list[-1]
    #     else:
    #         feature_data = fea
    #         edge_index = edge_list
    #     x = self.gcn(feature_data, edge_index)
    #
    #     return x

    # def pre_deal_feature(self, fea):
    #     feature_dict = dict()
    #     for feature_data in fea:
    #         # feature_data = feature_data.reshape(-1, feature_data.size(-1))
    #         # 先转置，得到不同属性对应所有节点的特征值
    #         feature_data = feature_data.transpose(0, 1)
    #
    #         num = feature_data.size(0)
    #         for i in range(num):
    #             target = feature_dict.get(i)
    #             if target is None:
    #                 feature_dict[i] = feature_data[i]
    #             else:
    #                 u_target = torch.cat([target, feature_data[i]], dim=-1)
    #                 feature_dict[i] = u_target
    #     feature = None
    #     for key in feature_dict.keys():  # 将list转换为Tensor表示
    #         tensor_item = feature_dict.get(key)
    #         feature_tensor = tensor_item.view(1, len(fea), -1)
    #         if feature is None:
    #             feature = feature_tensor
    #         else:
    #             feature = torch.cat([feature, feature_tensor], dim=0)
    #         # feature_dict[key] = feature_tensor
    #
    #     return feature

    # def forward(self, fea, idx_list, edge_list, test_mode=False):
    #     x = None
    #     prepare_feature_dict = self.pre_deal_feature(fea)
    #     for idx_key in prepare_feature_dict:
    #         feature = prepare_feature_dict.get(idx_key)
    #         h_t, (hn, cn) = self.lstm(feature)
    #         hn = hn.reshape(-1, self.hidden_dim)
    #         cn = hn.reshape(-1, self.hidden_dim)
    #         data = torch.cat([hn, cn], dim=1)
    #         data = self.mlp(data)
    #         # 更新节点特征作为下一次使用
    #         # self.feature_recent_dict[idx_key] = data
    #         # 更新到下一步进行图卷积的特征矩阵中
    #         data = data.reshape(data.size(-1), -1).to(self.device)
    #         if x is None:
    #             x = data
    #         else:
    #             x = torch.cat([x, data], dim=0)
    #     edge_index = edge_list[-1]
    #     x = self.gcn(x, edge_index)
    #
    #     return x

    def forward(self, fea, edge_list):
        result = list()
        # for feature_data, edge_index in zip(fea, edge_list):
        #     feature_data = feature_data.reshape(-1, feature_data.size(-1)).to(device=self.device)
        #     x = self.gcn(feature_data, edge_index)
        #     result.append(x)

        for feature_data in fea:
            x = feature_data.reshape(-1, feature_data.size(-1)).to(device=self.device)
            result.append(x)
        x = DataUtil.pre_deal_feature(result)
        # x = prepare_feature_dict.get(idx_key)
        # if self.cell is None:
        h_t, (hn, cn) = self.lstm(x)
        # else:
        #     h_t, self.cell = self.lstm(x, (self.cell[0].detach(), self.cell[1].detach()))

        hn = hn.reshape(-1, self.hidden_dim)
        # cn = hn.reshape(-1, self.hidden_dim)
        # x = torch.cat([h_t[-1]], dim=-1)
        out = self.mlp(hn)
        out = out.transpose(0, 1)
        out = self.gcn(out, edge_list[-1])
        # 更新节点特征作为下一次使用
        # self.feature_recent_dict[idx_key] = data
        # # 更新到下一步进行图卷积的特征矩阵中
        # x = x.transpose(0, 1)
        # if out is None:
        #     out = x
        # else:
        #     out = torch.cat([out, x], dim=0)
        out = out.transpose(0, 1)
        out = out.reshape(1, -1)
        # out = self.global_avg_pool(out)
        # out = out.transpose(0, 1)
        out = self.predict(out)

        return out

    def reset_weights(self):
        for m in self.modules():
            if not isinstance(m, (nn.ReLU, nn.ModuleList, nn.Dropout, AdaptiveAvgPool1d)):
                m.reset_parameters()

    def reset_parameters(self):
        pass

    def gcn(self, x, edge_index):

        x = self.lin(x)

        x = self.gcn_conv(x, edge_index)
        # x = self.rreLu(x)
        # x = self.reLu(x)
        x = self.gcn_conv(x, edge_index)
        x = self.cmlp(x)
        x = self.lin(x)
        return x

    def update_weight(self, new_weight):
        self.gcn_conv.lin.weight = nn.Parameter(new_weight)
        # self.weight = nn.Parameter(new_weight)

    def clear(self):
        self.feature_recent_dict.clear()
