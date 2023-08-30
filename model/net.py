"""Defines the neural network, losss function and metrics"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


class KRk_Net_ChatGPT(nn.Module):
    
    def __init__(self, params):
        super(KRk_Net_ChatGPT, self).__init__()

        self.num_channels = params.num_channels
        self.dropout_rate = params.dropout_rate

        self.fc1 = nn.Linear(self.num_channels, 256)  # Input layer to hidden layer
        self.fc2 = nn.Linear(256, 128)  # Hidden layer to hidden layer
        self.fc3 = nn.Linear(128, 32)   # Hidden layer to output layer

        self.dropout = nn.Dropout(self.dropout_rate)  # Dropout layer

    def forward(self, x):
        x = x.float()
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        batch_size = x.size(0)
        training_with_dropout = self.training and batch_size > 1 and self.dropout_rate > 0.0
        if training_with_dropout:
            x = self.dropout(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x
    
class KRk_Net_Bard(nn.Module):
    def __init__(self):
        super(KRk_Net_Bard, self).__init__()
        self.layer1 = nn.Linear(832, 128)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.layer3 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.layer4 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        batch_size = x.size(0)
        training_with_dropout = self.training and batch_size > 1 and self.dropout_rate > 0.0
        if training_with_dropout:
            x = self.dropout(x)
        x = self.layer3(x)
        x = self.relu(x)
        x = self.layer4(x)
        x = self.sigmoid(x) * 32
        return x