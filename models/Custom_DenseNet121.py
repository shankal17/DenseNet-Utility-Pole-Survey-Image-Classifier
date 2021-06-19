"""Classes of custom CNN image classifier

Classes
-------
CustomDenseNet121(num_classes)
    DenseNet121 with custom classifier
Classifier(num_classes)
    Custom classifier to match DenseNet121
"""

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
from collections import OrderedDict

class CustomDenseNet121(nn.Module):
    """DenseNet121 architecture with custom classifier
    ...
    
    Attributes
    ----------
    model : torchvision.models.densenet.DenseNet
        Pretrained DenseNet121 CNN

    Methods
    -------
    foward(x)
        Performs foward pass on input x
    """

    def __init__(self, num_classes):
        super(CustomDenseNet121, self).__init__()
        self.model = models.densenet121(pretrained=True)
        for i, param in enumerate(self.model.parameters()): # Freeze conv layers
            param.requires_grad = False

        # Replace DenseNet classifier with custom one
        self.model.classifier = Classifier(num_classes)
        for _, param in self.model.classifier.named_parameters():
            param.requires_grad = True

    def forward(self, x):
        """Performs foward pass on input x

        Parameters
        ----------
        x : torch.Tensor
            Network input
        
        Returns
        -------
        torch.Tensor
            Model output
        """

        out = self.model(x)
        return out

class Classifier(nn.Module):
    """Custom classifier to match DenseNet121
    ...
    
    Attributes
    ----------
    fc1 : torch.nn.Linear
        First fully connected layer
    fc2 : torch.nn.Linear
        Second fully connected layer
    fc3 : torch.nn.Linear
        Third fully connected layer
    fc4 : torch.nn.Linear
        Fourth fully connected layer
    dropout : nn.Dropout
        Dropout regularization

    Methods
    -------
    foward(x)
        Performs foward pass on input x
    """
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)
        
        # Dropout module with 0.3 drop probability
        self.dropout = nn.Dropout(p=0.3)
        
    def forward(self, x):
       
        # Now with dropout
        x = self.dropout(F.leaky_relu(self.fc1(x), 0.18))
        x = self.dropout(F.leaky_relu(self.fc2(x), 0.18))
        x = self.dropout(F.leaky_relu(self.fc3(x), 0.18))
        
        # output so no dropout here
        x = F.log_softmax(self.fc4(x), dim=1)
        
        return x

if __name__ == '__main__':
    model = CustomDenseNet121(2)
    print(model)
