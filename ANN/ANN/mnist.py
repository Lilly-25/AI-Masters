import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


def _get_conv_output_size(input_size, out_channels, model):
    batch_size = 1
    input_tensor = torch.autograd.Variable(torch.rand(batch_size, *input_size))
    output_feat = model.features(input_tensor)
    n_size = output_feat.data.view(batch_size, -1).size(1)
    return n_size


class MNIST(nn.Module):
    def __init__(self, out_channels):
        super(MNIST, self).__init__()

        conv_layers = []
        num_conv_layers = len(out_channels)

        for i in range(num_conv_layers):
            if i == 0:
                conv_layers.append(nn.Conv2d(1, out_channels[i], kernel_size=3, padding=1))
            else:
                conv_layers.append(nn.Conv2d(out_channels[i-1], out_channels[i], kernel_size=3, padding=1))

            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.features = nn.Sequential(*conv_layers)
        self.fc1 = nn.Linear(_get_conv_output_size((1, 28, 28), out_channels, self), 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def mnist_train(out_channels, num_epochs, batch_size, learning_rate, device, plots):
    # Set up transforms
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Load the FashionMNIST dataset
    train_dataset = datasets.MNIST(root='./mnist/train', train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST(root='./mnist/test', train=False, download=False, transform=transform)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    mnist_model = MNIST(out_channels)
    mnist_model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mnist_model.parameters(), lr=learning_rate)

    # Lists to track losses and accuracy
    train_losses = []
    test_losses = []
    accuracies = []

    # Training loop
    for epoch in range(num_epochs):
        # Train the model
        mnist_model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = mnist_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        accuracy = 100 * (correct / total)
        accuracies.append(accuracy)

        # Evaluate the model
        mnist_model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = mnist_model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * images.size(0)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}")

    if plots:
        # Plot the training and test losses
        plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
        plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # Plot the accuracy
        plt.plot(range(1, num_epochs + 1), accuracies, label='Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

    return mnist_model, (train_losses, test_losses, accuracies)


def mnist_apply(model, test_indexes):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    test_dataset = datasets.MNIST(root='./mnist/test', train=False, download=False, transform=transform)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    images = []
    labels_true = []
    for index in test_indexes:
        image, label = test_dataset[index]
        images.append(image)
        labels_true.append(label)

    images = torch.stack(images).to(device)
    labels_true = torch.tensor(labels_true).to(device)

    correct = 0
    with torch.no_grad():
        output = model(images)
        _, predicted = torch.max(output, 1)
        correct += (predicted == labels_true).sum().item()

    accuracy = 100 * correct / len(test_indexes)
    print('Accuracy of the network on the test images: %.2f %%' % accuracy)

    num_images = len(test_indexes)
    columns = min(5, num_images)
    rows = -(-num_images // columns)
    rows = min(rows, 2)  # Limit the maximum number of rows to 2

    fig_width = 2 * columns
    fig_height = 2 * rows
    fig, axes = plt.subplots(rows, columns, figsize=(fig_width, fig_height))

    for i, idx in enumerate(test_indexes[:10]):  # Print only the first 10 images
        image, label = test_dataset[idx]
        ax = axes[i // columns, i % columns]
        ax.imshow(image.squeeze(), cmap='gray')
        ax.set_title(f"Predicted: {predicted[i].item()}\nTrue: {label}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()
