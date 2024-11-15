import torch

from Controller import Controller

if __name__ == '__main__':
    # start = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Choose device based on availability
    # controller = Controller('opsahl-ucsocial/test', device)
    file_name = 'NEMData/network_1/no_repair/networking_1_modify_by_cgn__train_data__2024-06-24 125805.xlsx'
    adj_file = 'NEMData/network_1/no_repair/zx_network_1_dataset_adj_matrix.npy'
    feature_file = 'NEMData/network_1/no_repair/networking_1_modify_by_cgn__feature_data__2024-06-24 125805.xlsx'
    controller = Controller(file_name, device, adj_file, feature_file, "NEM")
    controller.clean_data()

