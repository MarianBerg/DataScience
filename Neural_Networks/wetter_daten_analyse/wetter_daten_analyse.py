import pandas as pd
import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt 

import time

import torch.nn as nn
import torch.optim as optim
import torch

from sklearn.model_selection import train_test_split

#Aufgabe
#
#Aufsetzen des Neural Networks
#
#Einige Grundparameter haben wir ja bereits festgelegt. Wir werden mit einem drei-schichtigen Neural Network arbeiten. 
#Da die Daten 784 Pixel haben, wird die Eingangsschicht 784 Neuronen bekommen. 
#Und da wir Zahlen zwischen 0 und 9 voneinander unterscheiden wollen, wird die Ausgabeschicht 10 Neuronen bekommen. 
#Die Anzahl der Neuronen im Hidden Layer setzen wir willkürlich auf 100. Ebenso willkürlich setzen wir die Lernrate auf 0.3. 
#Wie sich diese Werte auf die Performance des Neural Networks auswirken, werden wir später untersuchen. 
#
#    Erzeuge die Gewichts-Matrizen durch zufälliges Setzen von Werten zwischen -0.5 und 0.5. Verwende hierfür das numpy.random-Modul
#    Schreibe eine Funktion zur Signalübertragung im Neural Network. 
#Da wir sie später zum Testen des trainierten Neural Networks einsetzen werden, nennen wir die Funktion test(). 
#Als Argumente soll die Funktion den Input-Vektor und die beiden Gewichtsmatrizen enthalten.Zurückgeben soll die Funktion den Output-Vektor (ein 10-zeiliger Vektor).
#    Lasse dir mit print() den Output-Vektor vom ersten Element des Datensatzes ausgeben.

class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 4)
        self.fc6 = nn.Linear(4, 1)  # Assuming a regression problem
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        return x




def main():
    data = pd.read_csv('biking_one_hot.csv')
    

    # Separate features and target
    X = data.iloc[:, 1:].values  # All columns except the first one
    y = data.iloc[:, 0].values  # The first column
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

    from torch.utils.data import Dataset, DataLoader

    class CustomDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    # Create datasets
    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)

    batch_size = 100

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    # Initialize the network, loss function, and optimizer
    input_size = X_train.shape[1]
    model = SimpleNN(input_size)
    criterion = nn.MSELoss()  # Assuming a regression problem
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 300
    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.view(-1, 1))
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_loss = 0
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.view(-1, 1))
            test_loss += loss.item()

        test_loss /= len(test_loader)
        print(f'Test Loss: {test_loss:.4f}')


    torch.save(model.state_dict(), 'model_state_dict.pth')
    import shap

    # Initialize a SHAP explainer
    #explainer = shap.DeepExplainer(model, X_train)
#
    ## Calculate SHAP values for the test set
    #shap_values = explainer.shap_values(X_test)
#
    ## Summarize feature importance
    #shap.summary_plot(shap_values, X_test, feature_names=data.columns[1:])

    #
    #
#    test_data_label = label_data[3000:]
#    test_data_input = input_data[3000:]
#
#    test_data_input_normalized = np.asfarray(test_data_input) * 0.99 / 256 + 0.01
#
#    #Hyperparameter hier noch nicht benutzt
#    
#    input_size = 51
#    output_size = 1
#    #learning_rate = 0.0001
#    #num_epochs = 5
#    model = NN()
#    
#    input_weights = torch.tensor(np.random.uniform(-0.5, 0.5, (40, 51)), dtype=torch.float32)
#    output_weights = torch.tensor(np.random.uniform(-0.5, 0.5, (1, 40)), dtype=torch.float32)
#
#    
#
#    tensor_input = torch.tensor(train_data_input_normalized[0], dtype=torch.float32)
#
#    #adding batch dimension
#    tensor_input_batch = tensor_input.unsqueeze(0)
#    
#    output = model.test(tensor_input_batch, input_weights = input_weights, output_weights = output_weights)
#    
#    print("output: {} \n Shape: {}".format(output, output.shape))



if __name__ == "__main__":
    main()



