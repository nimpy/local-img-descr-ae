import torch.nn as nn
import torch.nn.functional as F
import torch

import sys
sys.path.append('/scratch/cloned_repositories/pytorch-msssim')
from pytorch_msssim import msssim


class AE(nn.Module):
    def __init__(self, latent_size=128, activation_str='elu', loss_str='bce'):
        super(AE, self).__init__()

        self.latent_size = latent_size

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

        self.fc_enc = nn.Linear(2048, latent_size)

        # decoder layers
        self.fc_dec = nn.Linear(latent_size, 2048)

        self.t_conv1 = nn.ConvTranspose2d(32, 32, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(32, 32, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(32, 1, 2, stride=2)

        if activation_str.lower() == 'elu':
            self.activation = F.elu
        elif activation_str.lower() == 'relu':
            self.activation = F.relu
        else:
            raise NotImplementedError

        if loss_str.lower() == 'bce':
            self.loss = nn.BCELoss()
        elif loss_str.lower() == 'msssim':
            self.loss = msssim
        else:
            raise NotImplementedError

    def encode(self, x):
        x = self.activation(self.conv1(self.zeropad1(x)))
        x = self.pool1(x)
        x = self.activation(self.conv2(self.zeropad2(x)))
        x = self.pool2(x)
        x = self.activation(self.conv3(self.zeropad3(x)))
        x = self.pool3(x)

        x = x.view(-1, 2048)
        x = self.fc_enc(x)

        return x

    def decode(self, x):
        x = self.fc_dec(x)
        x = x.view(-1, 32, 8, 8)

        x = self.activation(self.t_conv1(x))
        x = self.activation(self.t_conv2(x))
        x = torch.sigmoid(self.t_conv3(x))
        return x

    def forward(self, x):
        x = self.encode(x)
        # pdb.set_trace()
        x = self.decode(x)
        return x




