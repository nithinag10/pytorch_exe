import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data.dataloader
import   matplotlib.pyplot as plt

device = torch.device('cpu')

#HyperParameters
inputlayer = 784
hiddenlayer = 10
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

#loading MNIST Datasets
train_dataset = torchvision.datasets.MNIST(root='./data' , train=True , transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data' , train=False, transform=transforms.ToTensor(), download=False)

train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)

example = iter(train_dataloader)
sample, labels = example.next()

class NeuralNet(nn.Module):
    def __init__(self , inputlayer, hiddenlayer , num_classes):
        super(NeuralNet , self).__init__()
        self.linear1 = nn.Linear(inputlayer , hiddenlayer)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hiddenlayer , num_classes)

    def forward(self , x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out

model = NeuralNet(inputlayer , hiddenlayer , num_classes)

#loss and entrophy
criterian = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters() , lr=learning_rate)

#interate

n_total_steps = len(train_dataloader)

#training_loop
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_dataloader):
        images = images.reshape(-1, 784).to(device)
        labels = labels.to(device)
        output = model(images)
        loss = criterian(output , labels)

        #backward_propogation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1)%100 == 0:
            print(f'epoch {epoch+1}/{num_epochs} steps {i+1}/{n_total_steps} with loss {loss.item()}')


#testing

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_dataloader:
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        outputs = model(images)


        #torch.max returns value and index
        _, prediction = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (prediction == labels).sum().item()

    acc = 100.0 * n_correct / n_samples

    print(f'Accuracy of the model is {acc}.')
