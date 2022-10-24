import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, mode, image_size, num_classes):
        super(ConvNet, self).__init__()
        
        # Define various layers here, such as in the tutorial example
        # self.conv1 = nn.Conv2D(...)
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=40,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(1, 1))
        self.conv2 = nn.Conv2d(
            in_channels=40,
            out_channels=40,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(1, 1),
        )
        
        if mode == 1:
            self.fc1 = nn.Linear(image_size, 100)
            self.fc2 = nn.Linear(100, num_classes)
        if mode == 2 or mode == 3:
            self.fc1 = nn.Linear(19360, 100)
            self.fc2 = nn.Linear(100, num_classes)
        if mode == 4:
            self.fc1 = nn.Linear(23 * 23 * 40, 100)
            self.fc2 = nn.Linear(100, 100)
            self.fc3 = nn.Linear(100, num_classes)
        if mode == 5:
            self.fc1 = nn.Linear(23 * 23 * 40, 1000)
            self.fc2 = nn.Linear(1000, 1000)
            self.fc3 = nn.Linear(1000, num_classes)
            self.dropout = nn.Dropout(0.5)

        # This will select the forward pass function based on mode for the ConvNet.
        # Based on the question, you have 5 modes available for step 1 to 5.
        # During creation of each ConvNet model, you will assign one of the valid mode.
        # This will fix the forward function (and the network graph) for the entire training/testing
        if mode == 1:
            self.forward = self.model_1
        elif mode == 2:
            self.forward = self.model_2
        elif mode == 3:
            self.forward = self.model_3
        elif mode == 4:
            self.forward = self.model_4
        elif mode == 5:
            self.forward = self.model_5
        else: 
            print("Invalid mode ", mode, "selected. Select between 1-5")
            exit(0)
        
        
    # Baseline model. step 1
    def model_1(self, X):
        # ======================================================================
        # One fully connected layer. STEP 1: Create a fully connected (FC) hidden layer (with 100 neurons) with Sigmoid activation function.
        # Train it with SGD with a learning rate of 0.1 (a total of 60 epoch), a mini-batch size of 10, and no regularization.
        X = X.reshape(X.shape[0], -1)
        X = torch.sigmoid(self.fc1(X))
        X = self.fc2(X)

        return X

    # Use two convolutional layers.
    def model_2(self, X):
        # ======================================================================
        # Two convolutional layers + one fully connnected layer.
        X = torch.sigmoid(self.conv1(X))
        X = self.pool(X)
        X = torch.sigmoid(self.conv2(X))
        X = self.pool(X)
        X = X.reshape(X.shape[0], -1)
        X = torch.sigmoid(self.fc1(X))
        X = self.fc2(X)
        
        return X

    # Replace sigmoid with ReLU.
    def model_3(self, X):
        # ======================================================================
        # Two convolutional layers + one fully connected layer, with ReLU.
        X = F.relu(self.conv1(X))
        X = self.pool(X)
        X = F.relu(self.conv2(X))
        X = self.pool(X)
        X = X.reshape(X.shape[0], -1)
        X = F.relu(self.fc1(X))
        X = self.fc2(X)

        return X

    # Add one extra fully connected layer.
    def model_4(self, X):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        X = F.relu(self.conv1(X))
        X = self.pool(X)
        x = F.relu(self.conv2(X))
        X = self.pool(X)
        X = x.reshape(X.shape[0], -1)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)

        return X

    # Use Dropout now.
    def model_5(self, X):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        # and  + Dropout.
        X = F.relu(self.conv1(X))
        X = self.pool(X)
        x = F.relu(self.conv2(X))
        X = self.pool(X)
        X = x.reshape(X.shape[0], -1)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.dropout(X)
        X = self.fc3(X)

        return X
    
    
