import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

def show_images(img1, img2):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title("Orginal")
    ax2.set_title("Transformed")
    ax1.imshow(img1)
    ax2.imshow(img2)
    plt.show()

def show_image(img1):
    plt.imshow(img1)
    plt.show()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 6)
        self.conv2 = nn.Conv2d(32, 64, 6)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(p=0.3)
        self.f1 = nn.Linear(64 * 60 * 60, 1024)
        self.f2 = nn.Linear(1024, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.maxpool(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.maxpool(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.f1(x)
        x = F.relu(x)
        x = self.f2(x)
        x = F.log_softmax(x, dim=1)

        return x