import copy
import os
import random
import threading
import time

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from torch import nn

from Classifier import Classifier, Predict
from EvolGCN import EvolGCN
from util.CleanedDataSplit import CleanedDataSplit
from util.DataSplit import DataSplit
from util.DataUtil import DataUtil
from util.PlotUtil import PlotUtil
from util.TopologyAwareDataPreSplite import TopologyAwareDataSplite


class Controller:

    def __init__(self, file_name, device, adj_file=None, feature_file=None, mode="UCI"):
        self.device = device
        self.classifier_optimizer = None
        self.egcn_optimizer = None
        self.scheduler = None
        if mode == "UCI":
            self.ds = self.load_UCIsocial_data(file_name, self.device)
        elif mode == "OLD_FEATURE":
            self.ds = self.load_train_data(file_name, self.device, adj_file, feature_file)
        else:
            self.ds = TopologyAwareDataSplite(file_name=file_name, lstm_window=10, device="cpu",
                                              node_feature=4, edge_feature=1, label_class=1, adj_file=adj_file,
                                              node_num=18, edge_num=28,
                                              feature_file=feature_file)

        self.egcn = EvolGCN(self.ds.node_feature,
                            self.ds.node_feature, 64, 1, 1,
                            self.device, self.ds.node_num, self.ds.node_extra_feature).to(self.device)
        # 如果是链路预测，这里的参数又不一样
        # self.classifier = Classifier(2, 1).to(device)
        # self.predict = Predict(self.ds.node_num, 1).to(self.device)
        # 用于分类的损失函数
        self.loss_function = torch.nn.CrossEntropyLoss().to(self.device)
        # 用于回归的损失函数
        self.loss_MSE_function = torch.nn.MSELoss().to(self.device)
        self.edge_dict = dict()
        self.initial_state_dict = copy.deepcopy(self.egcn.state_dict())

    def load_UCIsocial_data(self, filename, device):
        ds = DataSplit(filename, skip_rows=2, device=device)
        return ds

    def load_train_data(self, filename, device, adj_file, feature_file):
        ds = DataSplit(file_name=filename, lstm_window=6, train=0.7, valid=0.2, device=device,
                       node_feature=3, edge_feature=2, label_class=1, adj_file=adj_file, node_num=18, edge_num=28,
                       feature_file=feature_file)

        return ds

    def use_cleaned_data_split(self, device, feature, adjs, label, extra=None):
        self.ds = CleanedDataSplit(lstm_window=6, train=0.7, valid=0.2, device=device,
                                   node_feature=2, edge_feature=2, label_class=1, node_num=18, edge_num=28,
                                   feature=feature, adjs=adjs, label=label, extra_info=extra)

    # def set_seed(self, seed=42):
    #     random.seed(seed)
    #     np.random.seed(seed)
    #     os.environ['PYTHONHASHSEED'] = str(seed)
    #     torch.manual_seed(seed)
    #     torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False

    # def train(self):
    #     self.init_optimizers()
    #     self.egcn.train()
    #     self.classifier.train()
    #     for epoch in range(5):
    #         self.edge_dict.clear()
    #         print('##############Epoch {:03d}#########'.format(epoch))
    #         # 每次更新参数前都梯度归零和初始化
    #         for s in self.ds.train_dataLoader:
    #             adj_sparse_list = self.ds.build_sparse_matrix(s.get("adj"))
    #             feature = s.get("feature")
    #             idx_list = s.get("feature_idx")
    #             label = s.get("label")
    #             node_mask = s.get("node_mask")
    #             label = label.reshape(-1, label.size(-1))
    #             node_feature = self.egcn(feature, idx_list, adj_sparse_list)
    #             edge_feature, mask = self.gather_node(node_feature, node_mask)
    #             out = self.classifier(edge_feature)
    #             loss = self.loss_function(out[mask], label[mask])
    #             loss.backward()
    #             self.optimizers_step()
    #
    #             print('Epoch {:03d} loss {:.4f}'.format(epoch, loss.item()))
    #
    #         self.save('./output/model_after_epoch_' + str(epoch) + '.pt')

    # 基于NEM提供的数据集业务的连通可用度预测
    def train_avail(self, lr, test_loss=False):

        plotUtil = PlotUtil([], [], "lr", "avg_loss")
        plt.ion()

        def run_thread(lrate, find_lr_mode=False):
            # total_test_num = 0
            if find_lr_mode:
                total_test_num = 1000
            else:
                total_test_num = 1

            for i in range(0, total_test_num):
                # 将模型权重设置为初始值，代替reset_weights()
                self.egcn.load_state_dict(self.initial_state_dict)
                # self.egcn.reset_weights()
                # print(self.egcn.state_dict())
                # torch.autograd.set_detect_anomaly(True)
                self.init_optimizers(lrate)
                print("lr = " + str(self.egcn_optimizer.param_groups[0]['lr']))
                lrate = lrate * 0.95
                skip = 5
                best_loss = float('inf')
                best_model_weight = None
                epoch_no_improve = 0
                num = 10000
                patience = 5
                for epoch in range(num):
                    self.egcn.train()
                    self.ds.reset()
                    # self.edge_dict.clear()
                    # print('##############Epoch {:03d}#########'.format(epoch))
                    # 每次更新参数前都梯度归零和初始化
                    mask_list = list()
                    out_list = list()
                    train_loss = 0.0
                    for s in self.ds.train_dataLoader:
                        adj_sparse_list = self.ds.build_sparse_matrix(s.get("adj"))
                        feature = s.get("feature")
                        mask = s.get("mask")[-1]
                        extra_feature = s.get("extra_feature")
                        out = self.egcn(feature, adj_sparse_list, extra_feature)
                        # out_list.append(out)
                        # mask_list.append(mask)
                        label = self.ds.label.to(self.device)
                        label = label.reshape(-1, 1)
                        # result = torch.cat(out_list).reshape(-1, 1)
                        # result.requires_grad_()
                        loss = self.loss_MSE_function(out, label[mask])
                        self.egcn_optimizer.zero_grad()
                        loss.backward()
                        self.optimizers_step()
                        train_loss += loss.item()
                    self.scheduler.step()

                    train_loss = train_loss / len(self.ds.train_dataLoader)

                    print('Epoch {:03d} loss {:.4f}'.format(epoch, train_loss))

                    # 先自己训练6次在验证
                    if epoch <= skip:
                        continue

                    self.egcn.eval()
                    out_list.clear()
                    mask_list.clear()
                    valid_loss = 0.0
                    with torch.no_grad():
                        for s in self.ds.valid_dataLoader:
                            adj_sparse_list = self.ds.build_sparse_matrix(s.get("adj"))
                            feature = s.get("feature")
                            mask = s.get("mask")[-1]
                            extra_feature = s.get("extra_feature")
                            out = self.egcn(feature, adj_sparse_list, extra_feature)
                            # out_list.append(out)
                            # mask_list.append(mask)
                            label = self.ds.valid_label.to(self.device)
                            label = label.reshape(-1, 1)
                            # result = torch.cat(out_list).reshape(-1, 1)
                            valid_loss = self.loss_MSE_function(out, label[mask])
                            valid_loss += valid_loss.item()

                        valid_loss = valid_loss / len(self.ds.valid_dataLoader)
                        print('Epoch {:03d} Training loss {:.6f} Valid loss {:.6f} Patience {:03d}'
                              .format(epoch, train_loss, valid_loss, patience - epoch_no_improve))
                        if best_loss > valid_loss:
                            best_loss = valid_loss
                            best_model_weight = self.egcn.state_dict()
                            epoch_no_improve = 0
                        else:
                            epoch_no_improve += 1

                        if epoch_no_improve == patience:
                            if find_lr_mode:
                                plotUtil.update_data(lrate, best_loss)
                                break

                            self.save('./output/model_after_epoch_' + str(num) + '.pt')
                            return

                    if best_model_weight is not None:
                        self.egcn.load_state_dict(best_model_weight)

        if test_loss:
            plotUtil.start_thread(function=run_thread, args=(lr, test_loss,))
            update_thread = threading.Thread(target=plotUtil.show_figure)
            update_thread.daemon = True
            update_thread.start()
            plt.show(block=True)
        else:
            run_thread(lr, test_loss)

    def init_optimizers(self, lr):
        # lr=0.01
        self.egcn_optimizer = torch.optim.Adamax(self.egcn.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.egcn_optimizer, gamma=0.9)

        # self.classifier_optimizer = torch.optim.Adam(self.classifier.parameters(), lr=0.01, weight_decay=1e-4)
        self.egcn_optimizer.zero_grad()
        # self.classifier_optimizer.zero_grad()
        self.egcn.clear()
        # 可靠度预测这是是reset,链路预测才是clear

    def optimizers_step(self):
        self.egcn_optimizer.step()
        # self.classifier_optimizer.step()
        # self.classifier_optimizer.zero_grad()
        self.ds.reset()

    def gather_node(self, node_feature, node_mask=None, test=False):
        p_len = int((self.ds.node_num * self.ds.node_num))
        e_predict = torch.zeros(p_len, 2, dtype=torch.float32)
        x_len = self.ds.node_num
        mask = list()
        if not test:
            for i in range(len(node_mask)):
                tmp = tuple(node_mask[i])
                node_mask[i] = tmp
        for x in range(x_len):
            for y in range(x_len):
                if y == x:
                    continue
                e_predict[x * self.ds.node_num + y] = node_feature[x]
                # e_predict[y * self.ds.node_num + x] = node_feature[y]
                if node_mask is not None and (x, y) in node_mask:
                    mask.append(x * self.ds.node_num + y)
                    self.edge_dict.setdefault((x, y), 1)
                    # mask.append(y * self.ds.node_num + x)

        # 生成负样本
        if not test:
            num = 0
            while True:
                idx = random.randint(0, int(self.ds.node_num) - 1)
                idy = random.randint(0, int(self.ds.node_num) - 1)

                if idx != idy and self.edge_dict.get((idx, idy)) is None:
                    num += 1
                    mask.append(idx * self.ds.node_num + idy)

                if num > 10:
                    break

        return e_predict.to(self.device), mask

    def save(self, model_save_path):
        torch.save(self.egcn.state_dict(), model_save_path)

    def test(self, model_file):
        self.egcn.eval()
        self.egcn.load_state_dict(torch.load(model_file))
        adj_list, feature_list, node_mask = self.ds.get_test_data_feature()
        adj_sparse_list = self.ds.build_sparse_matrix(adj_list, test_mode=True)
        pre = None
        adj = adj_sparse_list[0]
        feature = feature_list[0]
        softmax = nn.Softmax(dim=-1)
        # for adj, feature in zip(adj_sparse_list[0], feature_list):
        node_feature = self.egcn(feature, None, adj, test_mode=True)
        edge_feature, mask = self.gather_node(node_feature, node_mask, test=True)
        pre = softmax(edge_feature)
        result = pre.clone().detach()
        label = self.ds.label.reshape(-1, self.ds.label.size(-1))
        for i in range(result.size(0)):
            if result[i, 0] >= 0.5:
                result[i] = torch.IntTensor([1, 0])
            else:
                result[i] = torch.IntTensor([0, 1])
        self.calculate_accuracy(result, label, mask)

    def calculate_accuracy(self, pred, label, mask):
        correct = pred[mask].eq(label[mask]).sum().item() / 2
        acc = correct / int(len(mask))
        print('GCN Accuracy: {:.4f}'.format(acc))

    # 基于NEM提供的数据集业务的连通可用度预测
    def test_avail(self, model_file):
        self.egcn.eval()
        self.egcn.load_state_dict(torch.load(model_file))

        mask_list = list()
        out_list = list()
        for s in self.ds.test_dataLoader:
            adj_sparse_list = self.ds.build_sparse_matrix(s.get("adj"))
            feature = s.get("feature")
            # idx_list = s.get("feature_idx")
            # label = s.get("label")
            mask = s.get("mask")[-1]
            extra_feature = s.get("extra_feature")
            # label = label.reshape(label.size(-1), -1)
            out = self.egcn(feature, adj_sparse_list, extra_feature)
            # edge_feature, mask = self.gather_node(node_feature, node_mask)
            # node_feature = node_feature.view(1, self.ds.node_num)
            # out = self.predict(node_feature)
            out_list.append(out)
            mask_list.append(mask)
        label = self.ds.test_label
        label = label.reshape(label.size(-1), -1)
        result = torch.tensor(out_list).reshape(-1, 1).to(device=self.device).requires_grad_(False)
        writer = pd.ExcelWriter("./output/result_compare.xlsx")
        label = label[mask_list, :]
        df = pd.DataFrame({"result": result[:, 0], "label": label[:, 0]})
        df.to_excel(writer, index=False)
        writer.close()

    def clean_data(self, saved=True):
        outliers_list = list()
        feature, adjs = self.ds.get_all_data_sample(self.ds.data, self.ds.data.shape[0])
        data = DataUtil.pre_deal_feature(feature)
        y = self.ds.all_label.reshape(-1, 1)
        dbscan = DBSCAN(eps=0.2, min_samples=2)
        labels = dbscan.fit_predict(y)
        outliers_index = np.where(labels == -1)[0]
        outliers_list.extend(list(outliers_index))

        # for i in range(self.ds.node_feature):  # 节点数的总维度
        #     x = data[i].reshape(-1, data.size(-1))
        #     x = x.numpy()
        #     dbscan = DBSCAN(eps=0.8, min_samples=2)
        #     labels = dbscan.fit_predict(x)
        #     outliers_index = np.where(labels == -1)[0]
        #     outliers_list.extend(list(outliers_index))
        # 去重复
        outliers_list = list(set(outliers_list))
        new_label = np.delete(y, outliers_list).reshape(-1, 1)
        for index in sorted(outliers_list, reverse=True):
            del feature[index]
            del adjs[index]

        if saved:
            DataUtil.save_cleaned_data(feature, new_label, "./output/cleaned_data_" + str(time.time()) + ".xlsx")
            DataUtil.save_file_by_byte(adjs, "./output/cleaned_adj_" + str(time.time()) + ".xlsx")

        return feature, adjs, new_label
