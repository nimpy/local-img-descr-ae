import torch.nn as nn
import torch.nn.functional as F
import torch
import pdb

# TODO: add a fully-connected layer so that the architecture is the same as that of variational autoencoder
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        # encoder layers
        self.zeropad1 = nn.ZeroPad2d(1)
        self.conv1 = nn.Conv2d(1, 32, 3, padding=0)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.zeropad2 = nn.ZeroPad2d(1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=0)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.zeropad3 = nn.ZeroPad2d(1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=0)
        self.pool3 = nn.MaxPool2d(2, 2)

        # decoder layers
        self.t_conv1 = nn.ConvTranspose2d(32, 32, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(32, 32, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(32, 1, 2, stride=2)

    def encode(self, x):
        x = F.elu(self.conv1(self.zeropad1(x)))
        x = self.pool1(x)
        x = F.elu(self.conv2(self.zeropad2(x)))
        x = self.pool2(x)
        x = F.elu(self.conv3(self.zeropad3(x)))
        x = self.pool3(x)
        return x

    def decode(self, x):
        x = F.elu(self.t_conv1(x))
        x = F.elu(self.t_conv2(x))
        x = torch.sigmoid(self.t_conv3(x))
        return x

    def forward(self, x):
        x = self.encode(x)
        # pdb.set_trace()
        x = self.decode(x)
        return x

    loss = nn.BCELoss()



