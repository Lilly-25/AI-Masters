# importing the libraries
import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torch import optim
from torchvision import transforms

from torch.utils.data import random_split


import matplotlib.pyplot as plt 
# %matplotlib inline

# loading the dataset

train_set = datasets.FashionMNIST(root = './train' , train = True , download = True , transform= transforms.ToTensor())
test_set = datasets.FashionMNIST(root = './test' , train = False , download = True ,transform= transforms.ToTensor())


# splitting the train into train and val

train_set , val_set = random_split(train_set, lengths = [50000,10000])

# making the dataloader
batch_size = 32
train_loader = DataLoader(train_set , shuffle = True , batch_size = batch_size)
val_loader = DataLoader(val_set , shuffle = False , batch_size = batch_size) 
test_loader = DataLoader(test_set , shuffle = False , batch_size = batch_size)

# creating the neural_network architecture

class MLP(nn.Module):
    def __init__(self , input_size , hidden_dims):
        super(MLP,self).__init__()
        Layers = []
        Layers.append(nn.Linear(input_size, hidden_dims[0]))
        Layers.append(nn.ReLU())
        for i in range(0,len(hidden_dims)-1):
            Layers.append(nn.Linear(hidden_dims[i], hidden_dims[i]))
            Layers.append(nn.ReLU())
        Layers.append(nn.Linear(hidden_dims[-1], 10))

        self.model = nn.Sequential(*Layers) # using the * as we have to unpack the layers to be given to the sequential method
    
    def forward(self,x):
        return self.model(x)







def mlp_train(hidden_dims, epochs, batch_size, learning_rate, cuda, plots):

    # creating the data loaders with the batch_size 
    train_loader = DataLoader(train_set , shuffle = True , batch_size = batch_size)
    val_loader = DataLoader(val_set , shuffle = False , batch_size = batch_size) 
    test_loader = DataLoader(test_set , shuffle = False , batch_size = batch_size)
    
    input_size = 28*28

    # Initializing the model
    model = MLP(input_size , hidden_dims)

    # loss function

    loss_func = nn.CrossEntropyLoss()

    # optimizer
    optimizer = optim.Adam(model.parameters() , lr= learning_rate)

    # training and validation loops

    if torch.cuda.is_available():
        device = 'cuda'
    else :
        device = 'cpu'
    
    model.to(device)
    

    print(f"The Training is happening on : {device}")
    train_losses = []
    val_losses = []


    for epoch in range(epochs):
        train_loss = 0
        val_loss = 0
        model.train() # setting the model to be in training mode
        for batch in train_loader:
            optimizer.zero_grad()
            input , target = batch
            input = input.view(input.size(0),-1).to(device) # flattening the image 
            target = target.to(device)
            output = model(input)
            loss = loss_func(output , target)
            loss.backward()
            optimizer.step()

            train_loss += loss.data.item() * input.size(0)

            # total training loss for the epoch
        train_loss = train_loss / len(train_set)


        model.eval() # setting the model to be in validation or testing mode

        for batch in val_loader:
            input , target = batch
            input = input.view(input.size(0),-1).to(device)
            target = target.to(device)
            output = model(input)
            loss = loss_func(output , target)
            

            val_loss += loss.data.item() * input.size(0)

        val_loss = val_loss / len(val_set)

        train_losses.append(train_loss)
        val_losses.append(val_loss)


        # Print the training and testing loss for the epoch
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        

        
    #Plot the training and testing losses
    if plots== True:
        plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
        plt.plot(range(1, epochs+1), val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show() 
    
    return model , (train_losses , val_losses)


def mlp_apply(model, test_indexes):
    # Set model to evaluation mode for testing
    model.eval()
    
    if torch.cuda.is_available():
        device = 'cuda'
    else :
        device = 'cpu'
    
    # Get the selected test images and labels
    images = []
    labels_true = []
    for index in test_indexes:
        image, label = test_set[index]
        images.append(image)
        labels_true.append(label)
    
    # Convert images and labels to tensors
    images = torch.stack(images).to(device)
    labels_true = torch.tensor(labels_true).to(device)
    
    correct = 0
    # Perform inference
    with torch.no_grad():
        # Forward pass
        logits = model(images.view(-1, 28 * 28))
        _, predicted = torch.max(logits, dim=1)

        correct += (predicted == labels_true).sum().item()
    
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / len(test_indexes)))

    print()
    
    # Plot the images with true and predicted labels
    fig, axs = plt.subplots(2, 5, figsize=(12, 6))
    fig.suptitle('MLP Classification Results')
    for i, ax in enumerate(axs.flat):
        ax.imshow(images[i].cpu().numpy().squeeze(), cmap='gray')
        ax.axis('off')
        ax.set_title(f"True: {labels_true[i].item()}\nPred: {predicted[i].item()}")
    plt.show()