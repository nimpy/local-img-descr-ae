import torch.nn as nn
import torch.nn.functional as F
import torch
import pdb

import sys
sys.path.append('/scratch/cloned_repositories/pytorch-msssim')
from pytorch_msssim import msssim


class BetaVAE(nn.Module):
    def __init__(self, latent_size=128, activation_str='elu', loss_str='bce', beta=0.001):
        super(BetaVAE, self).__init__()

        self.latent_size = latent_size
        self.beta = beta

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

        self.fc_mu = nn.Linear(2048, latent_size)
        self.fc_var = nn.Linear(2048, latent_size)

        # decoder layers
        self.fc_z = nn.Linear(latent_size, 2048)
        self.t_conv1 = nn.ConvTranspose2d(32, 32, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(32, 32, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(32, 1, 2, stride=2)

        if activation_str.lower() == 'elu':
            self.activation = F.elu
        elif activation_str.lower() == 'relu':
            self.activation = F.relu
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

        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        # sample
        std = torch.exp(0.5 * logvar)  # e^(1/2 * log(std^2))
        eps = torch.randn_like(std)  # random ~ N(0, 1)
        z = eps.mul(std).add_(mu)

        return z, mu, logvar

    def decode(self, z):
        z = self.fc_z(z)
        z = z.view(-1, 32, 8, 8)

        rx = self.activation(self.t_conv1(z))
        rx = self.activation(self.t_conv2(rx))
        rx = torch.sigmoid(self.t_conv3(rx))

        return rx

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        rx = self.decode(z)
        return rx, mu, logvar

    def loss(self, recon_x, x, mu, logvar):

        # recon_loss1 = F.binary_cross_entropy(recon_x, x, reduction='sum')
        recon_loss2 = F.binary_cross_entropy(recon_x, x)

        # print()
        #             _                   _                             131072 == batch_size * h * w * c
        # print('    ', recon_loss1.item(), recon_loss2.item(), '      ', (recon_loss1 / recon_loss2).item())

        # kl_diverge1 = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_diverge2 = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)

        #             _                   _                             32 == batch_size
        # print('    ', kl_diverge1.item(), kl_diverge2.item(), '      ', (kl_diverge1 / kl_diverge2).item())


        # loss1 = (recon_loss1 + self.beta * kl_diverge1) / x.shape[0]
        loss2 = recon_loss2 + (self.beta * kl_diverge2) / 4096  # h * w * c

        return loss2
        # return (recon_loss + self.beta * kl_diverge) / x.shape[0]  # divide total loss by batch size
        # return recon_loss + self.beta * kl_diverge

