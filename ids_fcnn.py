#!/usr/bin/env python3

''' 
'''

# Import Numpy & PyTorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from ids_utils import *

### Re the architecture, it looks like you have a hidden layer (self.linear1). Just flagging in case you intended to first train with no hidden layer.

# Neural network matches the one in the part2-nn problem
class SimpleNet(nn.Module):
    # Initialize the layers
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(ids_features, ids_features)
        # self.relu1 = F.relu I believe this should be removed as the ReLU activation for layer 'linear1' is applied in the forward method.
        # Note this would be handled differently if you were using nn.Sequential to build the architecture. 
        self.linear2 = nn.Linear(ids_features, ids_classes)
   
    # Perform the computation
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x) # I believe you want to use F.relu(x) here.
        x = self.linear2(x) # Not sure, but I think a non-linearity activation function (e.g., x = F.sigmoid(x)) would be helpful. 
        # note, combining a sigmoid activation fxn in outer layer with cross entropy cost function helps to reduce neuron saturation/learning slowdown. 
        return x

# Define a utility function to train the model
def train(num_epochs, model, loss_fn, opt):
    for epoch in range(num_epochs):
        for xb,yb in train_dl:
            # Generate predictions
            pred = model(xb)
            loss = loss_fn(pred, yb)

            # Perform gradient descent
            loss.backward()
            opt.step()
            opt.zero_grad()
    print('Training loss: ', loss_fn(model(train_inputs), train_targets))


# Input data
df = ids_load_df_from_csv (outdir, file)
X_train, X_val, X_test, y_train, y_val, y_test = ids_split(df)

# convert from numpy arrays to torch tensors
train_inputs = torch.from_numpy(X_train.values).float()
train_targets = torch.from_numpy(y_train.values).float()

# Convert from torch tensors to torch dataset
train_ds = TensorDataset(train_inputs, train_targets)

# Define data loader
batch_size = 4
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

# Initialize model
model = SimpleNet()

# Initialize optimizer - to run gradient descent
# With batch size four, all four data points will be loaded per epoch
learning_rate = 0.001
opt = torch.optim.SGD(model.parameters(), learning_rate)

# Define loss function as mean squared error
loss_fn = F.mse_loss # I think cross_entropy may be better for classification. Per above, cross entropy works better with a sigmoid outer 
# layer activation (if you choose to try it) compared to a sigmoid outer layer with a MSE/quadratic cost function. 
# I believe F.cross_entropy(predictions, labels) prefers labels to be integers (e.g., 0,1,2,3,4,5,6,7,8,9 for MNIST).

# Record initial value of loss
loss = loss_fn(model(train_inputs), train_targets)
print ("Initial Loss:  ", loss)

# Train the model for 1500 epochs
num_epochs = 1500
train(num_epochs, model, loss_fn, opt)

# Generate predictions for the training set (no backpropagation)
train_preds = model(train_inputs)
print (train_preds)
