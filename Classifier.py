from torch import nn
from torch_geometric.nn import MLP


class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Classifier, self).__init__()
        self.dropout = nn.Dropout()
        self.lin1 = nn.Linear(input_dim, output_dim)
        # self.lin2 = nn.Linear(input_dim, num_class)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x = self.dropout(x)
        x = self.lin1(x)
        x = self.softmax(x)
        # x = self.lin2(x)
        return x


class Predict(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Predict, self).__init__()
        self.lin1 = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout()
        # self.lin2 = nn.Linear(input_dim, num_class)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.lin1(x)
        # x = self.dropout(x)
        x = self.relu(x)

        # x = self.lin2(x)
        return x
