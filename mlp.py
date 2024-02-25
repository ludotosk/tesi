import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import time
from sklearn.base import BaseEstimator

torch.manual_seed(12)

# Check for GPU availability and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Here I force the model to run on cpu since it's a small one is goes faster on a cpu then a gpu
device = 'cpu'

# how need to initialize a nn.relu each time you use it
class MLP(nn.Module, BaseEstimator):
    def __init__(self, epoch = 320, verbose = False, patience = 10, n_features = 79, out_neurons = 1): # out_neurons = 1 for binary classification, added for alowing multiclass classification
        super(MLP, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(n_features, 6), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(6, 3), nn.ReLU())
        if out_neurons == 1:
            self.fc3 = nn.Sequential(nn.Linear(3, out_neurons), nn.Sigmoid())
        else:
            self.fc3 = nn.Sequential(nn.Linear(3, out_neurons), nn.Softmax(dim=1))
        self.epoch = epoch
        self.verbose = verbose
        self.patience = patience
        self.estimator = "MultiLayerPerceptron"
        self.n_features = n_features
        self.out_neurons = out_neurons

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    
    def fit(self, X, y):
        # Initialize the model
        model = self.to(device)

        # Define the loss function and optimizer
        if self.out_neurons == 1:
            criterion = nn.BCELoss()
        else:
            criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Convert data to PyTorch tensors (assuming X and y are numpy arrays)
        X_tensor = torch.from_numpy(X.astype(np.float32)).to(device)
        if type(y) == pd.core.series.Series:
            y_tensor = torch.from_numpy(y.to_numpy())
        else:
            y_tensor = torch.from_numpy(y)

            # Convert y_tensor to long if doing multiclass classification
        if self.out_neurons > 1:
            y_tensor = y_tensor.long().to(device)
        else:
            y_tensor = y_tensor.float().to(device)


        # Train the model
        epochs = self.epoch
        batch_size = 200
        best_loss = np.inf
        no_improvement = 0  # Counter for epochs without improvement

        start_train = time.time()
        for epoch in range(epochs):
            for i in range(0, len(X_tensor), batch_size):
                inputs = X_tensor[i:i+batch_size]
                labels = y_tensor[i:i+batch_size]

                optimizer.zero_grad()

                outputs = model(inputs)
                if self.out_neurons == 1:
                    loss = criterion(outputs.squeeze(), labels)
                else:
                    loss = criterion(outputs, labels.squeeze())  # Remove extra dimension from labels
                loss.backward()
                optimizer.step()

            # Check for improvement
            if loss.item() < best_loss:
                best_loss = loss.item()
                no_improvement = 0
            else:
                no_improvement += 1

            # Optionally, print the loss at each epoch
            if self.verbose:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}")

            # If no improvement for 'patience' epochs, stop training
            if no_improvement >= self.patience:
                print("No improvement for {} epochs, stopping.".format(self.patience))
                break

        end_train = time.time()
        print("Execution time: ", end_train - start_train)

    def predict(self, X):
        # Convert data to PyTorch tensor
        X_tensor = torch.from_numpy(X.astype(np.float32)).to(device)

        # Run the model on the data
        with torch.no_grad():
            outputs = self.forward(X_tensor)

        # Apply a threshold to get boolean values
        predictions = outputs > 0.5

        if self.out_neurons > 1:
            # Get the predicted class number
            _, predicted_class = torch.max(outputs, 1)

            # Convert predictions back to numpy and return
            return predicted_class.cpu().numpy()
        else:
            # Convert predictions back to numpy and return
            return predictions.squeeze().cpu().numpy()