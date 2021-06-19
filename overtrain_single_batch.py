"""Functions used to overtrain a single batch of data

This module contains the function to overtrain a model to
a single batch of data

Functions
---------
overfit(model, train_loader, epochs=50, plot_name=None, log_name=None)
    Overfits model to single batch of data
"""

import logging
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm
from torchvision import transforms, datasets
from models.Custom_DenseNet121 import CustomDenseNet121

def overfit(model, train_loader, epochs=50, plot_name=None, log_name=None):
    """Overfits model to single batch of data

    Parameters
    ----------
    model : torch.nn.Network
        Model to be trained
    train_loader : torch.utils.data.DataLoader
        Data to which the model will be fit
    epochs : int, optional
        Number of training epochs
    plot_name : string, optional
        Name to name the loss vs epoch graph
    log_name : string, optional
        Name or path of the loss log file
    """

    if log_name:
        perform_logger = logging.getLogger()
        perform_logger.setLevel(logging.DEBUG)

        output_file_handler = logging.FileHandler(log_name)
        stdout_handler = logging.StreamHandler(sys.stdout)

        perform_logger.addHandler(output_file_handler)
        perform_logger.addHandler(stdout_handler)
    
    # Use GPU if it's available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.model.classifier.parameters(), lr=0.003)
    model.to(device)

    dataiter = iter(trainloader)
    inputs, labels = dataiter.next()

    running_loss = 0
    losses = []
    for epoch in range(epochs):
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy = torch.mean(equals.type(torch.FloatTensor)).item()

        losses.append(loss.item())
        if log_name:
            perform_logger.debug(f"epoch {epoch}.."
                f"loss {loss.item()}.."
                f"Accuracy: {accuracy}..")

    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    if plot_name:
        plt.savefig(plot_name)
    plt.show()


if __name__ == '__main__':
    data_dir = 'data/processed/train_val_split'

    train_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

    model = CustomDenseNet121(len(train_data.classes))

    overfit(model, trainloader, epochs=50, plot_name='overfit_single_batch.png', log_name='overfit_loss_accuracy.log')