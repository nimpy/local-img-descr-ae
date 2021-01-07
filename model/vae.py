import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import pdb


class BetaVAE(nn.Module):
    def __init__(self, latent_size=128, beta=0.001):
        super(BetaVAE, self).__init__()

        self.latent_size = latent_size
        self.beta = beta

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

        self.t_conv1 = nn.ConvTranspose2d(32, 32, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(32, 32, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(32, 1, 2, stride=2)

        self.fc_z = nn.Linear(latent_size, 2048)


    def encode(self, x):
        x = F.elu(self.conv1(self.zeropad1(x)))
        x = self.pool1(x)
        x = F.elu(self.conv2(self.zeropad2(x)))
        x = self.pool2(x)
        x = F.elu(self.conv3(self.zeropad3(x)))
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

        rx = F.elu(self.t_conv1(z))
        rx = F.elu(self.t_conv2(rx))
        rx = torch.sigmoid(self.t_conv3(rx))

        return rx

    def forward(self, x):
        # x = F.elu(self.conv1(self.zeropad1(x)))
        # x = self.pool1(x)
        # x = F.elu(self.conv2(self.zeropad2(x)))
        # x = self.pool2(x)
        # x = F.elu(self.conv3(self.zeropad3(x)))
        # x = self.pool3(x)
        #
        # x = x.view(-1, 2048)
        #
        # mu = self.fc_mu(x)
        # logvar = self.fc_var(x)
        #
        # # sample
        # std = torch.exp(0.5 * logvar)  # e^(1/2 * log(std^2))
        # eps = torch.randn_like(std)  # random ~ N(0, 1)
        # z = eps.mul(std).add_(mu)

        z, mu, logvar = self.encode(x)

        # z = self.fc_z(z)
        # z = z.view(-1, 32, 8, 8)
        #
        # rx = F.elu(self.t_conv1(z))
        # rx = F.elu(self.t_conv2(rx))
        # rx = torch.sigmoid(self.t_conv3(rx))

        rx = self.decode(z)

        return rx, mu, logvar

    def loss(self, recon_x, x, mu, logvar):
        # reconstruction losses are summed over all elements and batch
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_diverge = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # print(self.beta, recon_loss.item() / x.shape[0], (self.beta * kl_diverge.item()) / x.shape[0])

        return (recon_loss + self.beta * kl_diverge) / x.shape[0]  # divide total loss by batch size

