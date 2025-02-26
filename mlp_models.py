import torch
import torch.nn as nn
import torch.optim as optim

'''
Linear model:
Input dimension: 2304
Output dimension: 2

'''


'''
SAPLMA’s classifier employs a feedforward
neural network, featuring three hidden layers with
decreasing numbers of hidden units (256, 128, 64),
all utilizing ReLU activations. The final layer is a
sigmoid output. We use the Adam optimizer. We
do not fine-tune any of these hyper-parameters for
this task. The classifier is trained for 5 epochs

Input dimension: 2304
Output dimension: 2


'''

import torch
import torch.nn as nn
import torch.optim as optim

# Define the feedforward neural network model
class SAPLMANet(nn.Module):
    def __init__(self, input_dim=2304):
        super(SAPLMANet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),   # final layer outputting a single value
            nn.Sigmoid()        # sigmoid activation for binary classification
        )
        self.loss = nn.BCELoss()
        

    def forward(self, state, label):
        out = self.net(state)
        loss = self.loss(out, label)
        return out, loss

class SimpleNet(nn.Module):
    def __init__(self, input_dim=2304):
        super(SimpleNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2),
            nn.ReLU(),
            nn.Linear(2, 1),   # final layer outputting a single value
            nn.Sigmoid()        # sigmoid activation for binary classification
        )
        self.loss = nn.BCELoss()

    def forward(self, state, label):
        out = self.net(state)
        loss = self.loss(out, label)
        return out, loss

class DimMapNet(nn.Module):
    def __init__(self, input_dim=2304):
        super(DimMapNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 1),   # final layer outputting a single value
            nn.Sigmoid()        # sigmoid activation for binary classification
        )
        self.loss = nn.BCELoss()

    def forward(self, state, label):
        out = self.net(state)
        loss = self.loss(out, label)
        return out, loss
