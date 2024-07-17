import math
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.datasets import Planetoid

from Classifier import Classifier
from Controller import Controller
from EvolGCN import EvolGCN
from util.DataSplit import DataSplit


def load_Cora_data(name, device):
    dataset = Planetoid(root='./' + name + '/', name=name)
    data = dataset[0].to(device)
    num_node_features = dataset.num_node_features
    return data, num_node_features, dataset.num_classes


def load_UCIsocial_data(filename, device):
    ds = DataSplit(filename, skip_rows=2, device=device)
    return ds


def test(model, classifier, data):
    model.eval()
    classifier.eval()
    out = model(data.x, data.edge_index)
    _, pred = classifier(out).max(dim=1)
    correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / int(data.test_mask.sum())
    print('GCN Accuracy: {:.4f}'.format(acc))


# def train(model, classifier, data):
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
#     loss_function = torch.nn.CrossEntropyLoss().to(device)
#     model.train()
#     for epoch in range(200):
#         # 每次更新参数前都梯度归零和初始化
#         optimizer.zero_grad()
#         out = model(data.x, data.edge_index)
#         out = classifier(out)
#         optimizer.zero_grad()
#         loss = loss_function(out[data.train_mask], data.y[data.train_mask])
#         loss.backward()
#         optimizer.step()
#
#         print('Epoch {:03d} loss {:.4f}'.format(epoch, loss.item()))


def gather_node(adj_sparse, node_feature):
    e_feature_list = list()
    for indices in adj_sparse.indices():
        edge_feature = node_feature[indices]
        e_feature_list.append(edge_feature)
    return torch.cat(e_feature_list, dim=1)


def run_epocch(model, classifier, ds, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-4)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    model.train()
    for epoch in range(10):
        print('##############Epoch {:03d}#########'.format(epoch))
        # 每次更新参数前都梯度归零和初始化
        optimizer.zero_grad()
        model.clear()
        for s in ds.train_dataLoader:
            adj_sparse_list = ds.build_sparse_matrix(s.get("adj"))
            # feature = feature.view(1, -1, -1) #lstm batch_first
            feature = s.get("feature")
            idx_list = s.get("feature_idx")
            node_feature = model(feature, idx_list, adj_sparse_list)
            edge_feature = gather_node(adj_sparse_list[-1], node_feature)
            out = classifier(edge_feature)
            # loss = loss_function(out[data.train_mask], data.y[data.train_mask])
            # loss.backward()
            # optimizer.step()

        # print('Epoch {:03d} loss {:.4f}'.format(epoch, loss.item()))


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    set_seed(3047)
    # start = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Choose device based on availability
    # controller = Controller('opsahl-ucsocial/test', device)
    # file_name = 'NEMData/network_1/no_repair/networking_1_modify_by_cgn__train_data__2024-06-25 214959.xlsx'
    file_name = 'NEMData/network_1/repair/networking_1_modify_by_cgn__train_data__with_sequence.xlsx'
    adj_file = 'NEMData/network_1/no_repair/zx_network_1_dataset_adj_matrix.npy'
    feature_file = 'NEMData/network_1/no_repair/networking_1_modify_by_cgn__feature_data__2024-06-24 205501.xlsx'

    controller = Controller(file_name, device, adj_file, feature_file, "NEM")
    # controller.train()
    feature, adjs, label = controller.clean_data(saved=False)
    controller.use_cleaned_data_split(device, feature, adjs, label, controller.ds.sequence_info)
    # 备选1：3.5249975523875253e-06 备选2：0.00463291230159753
    controller.train_avail(test_loss=False, lr=0.01)
    # data, num_node_features, num_classes = load_data('Cora', device)
    # ds = load_UCIsocial_data('opsahl-ucsocial/out.opsahl-ucsocial', device)
    # ds = load_UCIsocial_data('opsahl-ucsocial/test', device)
    # egcn = EvolGCN(ds.node_feature, ds.node_feature, 64, 1, device).to(device)
    # classifier = Classifier(ds.edge_feature, 2).to(device)
    # run_epocch(egcn, classifier, ds, device)

    # x = torch.randn(5, 16, device=device)  # Example node features
    # edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4],
    #                            [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long, device=device)
    # train(egcn, classifier, data)
    # test(egcn, classifier, data)
