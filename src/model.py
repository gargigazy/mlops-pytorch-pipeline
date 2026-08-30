# model definition for the image classifier
# using resnet18 from torchvision because its easy to use and works ok on cifar10

import torch.nn as nn
from torchvision import models


class SimpleCNN(nn.Module):
    # backup model incase resnet is too heavy, a basic cnn
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_model(architecture="resnet18", num_classes=10):
    # returns the model based on the name from the config file
    if architecture == "resnet18":
        model = models.resnet18(weights=None)
        # cifar10 images are only 32x32 so the default 7x7 conv is too big
        # found this trick online, works better for small images
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        # replace the last layer for 10 classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif architecture == "simple_cnn":
        model = SimpleCNN(num_classes)
    else:
        raise ValueError("unknown architecture: " + architecture)
    return model
