from nn import *
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np


torch.manual_seed(0)

train_data = torch.zeros((1000, 13))
train_labels = torch.zeros((1000, 1))
training_dataset = TensorDataset(train_data, train_labels)

model = NeuralNetwork(fc_nodes_list=[100, 100, 100], input_dim=13)
loss_fn = nn.MSELoss()

train_loader = DataLoader(training_dataset, shuffle=True, num_workers=1)

model.train_model(train_loader, loss_fn, init_lr=0.001, num_epochs=10)