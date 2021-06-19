"""Functions used to train and tune the configured hyperparameters

This module contains the functions needed to train and tune the
defined model given the domain of possible hyperparameter values.

Functions
---------
load_data(data_dir='/.data')
    Loads the training, validation, and test datasets.
train_classifier(config, checkpoint_dir=None, data_dir=None, num_classes=6)
    Trains the defined classifer using the specified hyperparameters
main(num_samples=2, max_num_epochs=10, gpus_per_trial=1, data_dir='/.data')
    Tunes hyperparameters using specified training parameter
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm
from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from torchvision import transforms, datasets
from models.Custom_DenseNet121 import CustomDenseNet121

def load_data(data_dir='/.data'):
    """Loads the training, validation, and test datasets

    Parameters
    ----------
    data_dir : string, optional
        Directory path to split dataset

    Returns
    -------
    datasets.ImageFolder
        Training dataset
    datasets.ImageFolder
        Validation dataset
    datasets.ImageFolder
        Testing dataset
    """
    train_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    val_data = datasets.ImageFolder(data_dir + '/val', transform=test_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
    
    return train_data, val_data, test_data

def train_classifier(config, checkpoint_dir=None, data_dir=None):
    """Trains the defined classifer using the specified hyperparameters

    Parameters
    ----------
    config: dict
        Dictionary of hyperparameter values
    checkpiont_dir : string, optional
        Path to model checkpoints
    data_dir : string, optional
        Directory path to split dataset
    """

    # Load the datasets
    train_data, val_data, test_data = load_data(data_dir)
    num_classes = len(val_data.classes)

    # Use GPU if it's available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define CNN (can be made configurable if desired)
    model = CustomDenseNet121(num_classes)
    model.to(device)

    # Define loss criterion
    criterion = nn.NLLLoss()

    # Define optimizer so that it's configurable
    optimizer = optim.Adam(model.model.classifier.parameters(), lr=config['lr'])

    # Load model checkpoint if available
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    # Create configurable dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=int(config['batch_size']), shuffle=True, num_workers=4)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=int(config['batch_size']), shuffle=True, num_workers=4)

    for epoch in range(15):
        running_loss = 0.0
        steps = 0
        model.train()
        for i, data in enumerate(trainloader, 0): # Loop over training dataset
            inputs, labels = data

            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero grads
            optimizer.zero_grad()
            
            # Forward pass and calculate loss
            logps = model.forward(inputs)
            loss = criterion(logps, labels)

            # Backpropagate errors
            loss.backward()

            # Optimize step
            optimizer.step()

            running_loss += loss.item()
            steps += 1
            if i % 200 == 199:
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / steps))
                running_loss = 0.0
        
        # Validation step
        val_loss = 0.0
        accuracy = 0

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(valloader, 0): # Run through valadation dataset
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model.forward(inputs)
                batch_loss = criterion(logps, labels)
                
                val_loss += batch_loss.item()
                
                # Calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
        # Save model checkpoint
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        # Report performance data to ray tune
        tune.report(loss = (val_loss/len(valloader)), accuracy=(accuracy/len(valloader)))

def main(num_samples=2, max_num_epochs=10, gpus_per_trial=1, data_dir='/.data'):
    """Tunes hyperparameters using specified training parameter

    Parameters
    ----------
    num_samples : int, optional
        Number of hyperparameter combinations to try
    max_num_epochs : int, optional
        Maximum number of epochs if Tune doesn't stop it early
    gpus_per_trial : int, optional
        Number of gpus to split the samples
    data_dir : string, optional
        Directory path to split dataset
    """

    load_data(data_dir)

    # Define hyperparameter search space
    config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([8, 16, 32])
    }

    # Define hyperparam optim algorithm
    # This method can be found here: https://arxiv.org/abs/1810.05934
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    
    # Configure metrics for Ray Tune to report
    reporter = CLIReporter(
        metric_columns=["loss", "accuracy", "training_iteration"])
    
    # Execute the training with hyperparameter tuning as defined
    result = tune.run(
        partial(train_classifier, data_dir=data_dir),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)
    
    # Find the model and configuration that gave the best results
    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))


if __name__ == '__main__':

    # Specify the data directory path
    data_dir = 'C:/Users/17175/Documents/Pole_Image_Classifier_V2/data/processed/train_val_split'
    
    # Perform training/hyperparameter tuning
    main(num_samples=30, max_num_epochs=20, data_dir=data_dir)
