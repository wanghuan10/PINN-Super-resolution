import torch
import torch.nn as nn
import numpy as np

# torch.backends.cuda.matmul.allow_tf32 = (
#     False  # This is for Nvidia Ampere GPU Architechture
# )
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.manual_seed(1234)
np.random.seed(1234)

class layer(nn.Module):
    def __init__(self, n_in, n_out, activation):
        super().__init__()
        self.layer = nn.Linear(n_in, n_out)
        self.activation = activation

    def forward(self, x):
        x = self.layer(x)
        if self.activation:
            x = self.activation(x)
        return x

class DNN(nn.Module):
    def __init__(self, layers, ub, lb, activation=nn.Tanh()):
        super().__init__()
        self.net = nn.ModuleList()
        num_layers = len(layers)
        for l in range(0, num_layers - 2):
            self.net.append(layer(layers[l], layers[l + 1], activation))
        self.net.append(layer(layers[-2], layers[-1], activation=None))
        self.ub = torch.tensor(ub, dtype=torch.float).to(device)
        self.lb = torch.tensor(lb, dtype=torch.float).to(device)
        self.net.apply(weights_init)  # xavier initialization

    def forward(self, x):
        # [0, 1]归一化
        x = (x - self.lb) / (self.ub - self.lb)  # Min-max scaling
        out = x
        # 正态分布归一化
        # x_mean = x.mean(axis=0)
        # x_var = x.var(axis=0)
        # x = (x - x_mean) / (torch.sqrt(x_var + 1e-5))
        # out = x
        for layer in self.net:
            out = layer(out)
        return out


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.zeros_(m.bias.data)

