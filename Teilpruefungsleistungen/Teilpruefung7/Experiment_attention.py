import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transform
import numpy as np
import matplotlib.pyplot as plt


class NN(nn.Module):
    def __init__(self, input_neurons, output_neurons):
        super(NN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(784, input_neurons),
            nn.ReLU(),
            nn.Linear(input_neurons, 512),
            nn.ReLU(),
            nn.Linear(512, output_neurons),
        )
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        logits = self.network(x)
        return logits

class Data(Dataset):

    def __init__(self):
        data = np.loadtxt("mnist_data.csv", delimiter = ",", dtype = np.float32, skiprows = 1)

        self.labels = torch.from_numpy(data[:, 0])
        self.images = torch.from_numpy(data[:, 1:])
        self.n_samples = data.shape[0]

    def __getitem__(self, index):
        return self.images[index], self.labels[index]
    
        

    def __len__(self):
        return self.n_samples
    


#Runtime starten
start = time.time()

dataset = Data()
train_size  = int(0.8 * len(dataset))
test_size   = len(dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(dataset = train_dataset, shuffle = True)
test_loader  = DataLoader(dataset = test_dataset, shuffle = True)



model = NN(212, 10)

input_size = 784 #Anzahl Pixel
output_size = 10 #Die 10 Klassen der Daten
learning_rate = 0.0001
num_epochs = 5

loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = learning_rate)

#Netzwerk Training:
for epoch in range(0, num_epochs):
    for idx, (image, label) in enumerate(train_loader):

        label = label.type(torch.LongTensor)

        #inputs in das netz schicken
        output = model(image)
        loss = loss_function(output, label)

        #Backpropagation
        optimizer.zero_grad()   #gradient auf 0 setzen
        loss.backward()
        optimizer.step()

        if idx % 100 == 0:
            loss, current = loss.item(), idx * len(image)
            print(f"loss: {loss} Durchlauf: {current}")

#Netzwerk Performance Test:
richtige = []
for idx, (image, label) in enumerate(test_loader):

    label = label.type(torch.LongTensor)

    #Input ins Netzwerk:
      
    output = model(image)
    output = output.argmax() 

    if label == output:
        richtige.append(1) 
    else:
        richtige.append(0)


performance = sum(richtige)/test_size

with open('results.txt', 'a') as file:
    # Write the results to the file
    file.write("\nErgebnisse ELU:\n")
    file.write(f"test_size: {test_size}\n")

    file.write(f"Richtige Insgesamt: {sum(richtige)}\n")
    file.write(f"Performance: {performance}\n")
    file.write(f"Learning_rate: {learning_rate}\n") 

# Runtime messen
end = time.time()
print("Runtime: {} sec".format(end-start))

