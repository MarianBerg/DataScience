import torch
import torch.nn as nn
import numpy as np

class NN(nn.Module):
    def __init__(self, anzahl_neuronen_eingabeschicht, anzahl_neuronen_ausgabeschicht):
        super(NN, self).__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(28*28, anzahl_neuronen_eingabeschicht),                           # zero layer
            nn.Sigmoid(),                                                               # first_layer
            nn.Linear(anzahl_neuronen_eingabeschicht, anzahl_neuronen_ausgabeschicht),  # second_layer
            nn.Sigmoid(),                                                               # third_layer    
        )

    def forward(self, x):
        x = self.flatten(x)
        output = self.network(x)
        return output
    
    def train(self, input_list, target_list, learning_rate):
        # Convert input and target lists to numpy arrays
        input_vector = np.array(input_list, ndmin=2)
        targets = np.array(target_list, ndmin=2)
        
        # Convert numpy arrays to tensors
        input_tensor = torch.tensor(input_vector, dtype=torch.float32)
        targets_tensor = torch.tensor(targets, dtype=torch.float32)
        
        # Forward pass
        zero_layer_output = self.network[0](input_tensor)
        first_layer_output = self.network[1](zero_layer_output)
        second_layer_output = self.network[2](first_layer_output)
        third_layer_output = self.network[3](second_layer_output)
        
        # Compute the error
        output_errors = (targets_tensor - third_layer_output).pow(2).mean()
        
        # Backpropagation
        # Compute gradients for the second layer weights
        second_layer_delta = (targets_tensor - third_layer_output) * third_layer_output * (1 - third_layer_output)
        d_w_second_layer = torch.matmul(second_layer_delta.T, first_layer_output)
        
        # Update second layer weights
        with torch.no_grad():
            self.network[2].weight -= learning_rate * d_w_second_layer
            self.network[2].bias -= learning_rate * second_layer_delta.mean(axis=0)
        
        # Compute gradients for the zero layer weights
        first_layer_delta = torch.matmul(second_layer_delta, self.network[2].weight.T) * first_layer_output * (1 - first_layer_output)
        d_w_zero_layer = torch.matmul(first_layer_delta.T, input_tensor)
        
        # Update zero layer weights
        with torch.no_grad():
            self.network[0].weight -= learning_rate * d_w_zero_layer
            self.network[0].bias -= learning_rate * first_layer_delta.mean(axis=0)
        
        return output_errors.item()

# Initialize the model
model = NN(100, 10)

# Example input and target lists
input_list = [0.1] * (28*28)  # Example input, replace with your actual data
target_list = [0] * 10  # Example target, replace with your actual data

# Training parameters
learning_rate = 0.01

# Train the model
loss = model.train(input_list, target_list, learning_rate)
print("Training loss:", loss)
