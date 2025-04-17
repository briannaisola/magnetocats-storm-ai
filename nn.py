import torch
from torch import nn
import torch.optim as optim
import tqdm


class NeuralNetwork(nn.Module):
    def __init__(self, fc_nodes_list: list, input_dim=1):
        super().__init__()
        
        assert len(fc_nodes_list) > 2, "`fc_nodes_list` must have at least 3 elements, starting with the input layer but not the output layer"
        self.fc_nodes_list = fc_nodes_list
        self.input_dim = input_dim
        self.module_list = nn.ModuleList()
        # Input layer
        self.module_list.append(nn.Linear(self.input_dim, fc_nodes_list[0]))

        # Hidden layers
        for layer_num in range(1, len(fc_nodes_list) - 1):
            self.module_list.append(nn.Linear(fc_nodes_list[layer_num], fc_nodes_list[layer_num + 1]))
            self.module_list.append(nn.ReLU())
        
        # Output layer
        self.module_list.append(nn.Linear(fc_nodes_list[-1], 1))

    def forward(self, x):
        for module in self.module_list:
            x = module(x)
        
        return x
    
    
    def train(train_loader, loss_fn, init_lr, num_epochs=200):
        optimizer = optim.Adam(self.parameters(), lr=init_lr)
        model.train()  # Put the model in training mode
        for epoch in tqdm.trange(num_epochs, desc="Training model. Epoch"):
            for step_num, datum in enumerate(train_loader):  # Objects in a dataloader must be accessed this way
                inputs, targets = datum
                optimizer.zero_grad()  # Reset the gradients so they don't accumulate over successive backpropagations
                outputs = self(inputs)  # Good ol' forward pass
                loss = loss_fn(outputs, targets)  # Calculate the loss between predicted data and ground truth
                loss.backward()  # Use loss to update weights
                optimizer.step()  # Update the stochastic gradient descent optimizer
