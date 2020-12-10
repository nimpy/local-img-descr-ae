import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import pdb


class BetaVAE(nn.Module):
    def __init__(self, latent_size=128, beta=1):
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


        ####
        # self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2)  # TODO for all conv add batch norm
        # self.conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2)
        # self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # self.conv4 = nn.Conv2d(64, 64, kernel_size=4, stride=2)
        #
        # self.fc_mu = nn.Linear(256, latent_size)
        # self.fc_var = nn.Linear(256, latent_size)
        #
        # self.t_conv1 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, output_padding=0)  # TODO batch norm?
        # self.t_conv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, output_padding=0)
        # self.t_conv3 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, output_padding=0)
        # self.t_conv4 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, output_padding=0)
        #
        # self.fc_z = nn.Linear(latent_size, 256)

    def forward(self, x):

        x = F.elu(self.conv1(self.zeropad1(x)))
        x = self.pool1(x)
        x = F.elu(self.conv2(self.zeropad2(x)))
        x = self.pool2(x)
        x = F.elu(self.conv3(self.zeropad3(x)))
        x = self.pool3(x)  # compressed representation

        x = x.view(-1, 2048)

        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        # sample
        std = torch.exp(0.5 * logvar)  # e^(1/2 * log(std^2))
        eps = torch.randn_like(std)  # random ~ N(0, 1)
        z = eps.mul(std).add_(mu)
        z = self.fc_z(z)
        z = z.view(-1, 32, 8, 8)


        rx = F.elu(self.t_conv1(z))
        rx = F.elu(self.t_conv2(rx))
        rx = torch.sigmoid(self.t_conv3(rx))


        #####

        # x = F.elu(self.conv1(x))
        # x = F.elu(self.conv2(x))
        # x = F.elu(self.conv3(x))
        # x = F.elu(self.conv4(x))
        #
        # x = x.view(-1, 256)
        #
        # mu = self.fc_mu(x)
        # logvar = self.fc_var(x)
        #
        # # sample
        # std = torch.exp(0.5 * logvar)  # e^(1/2 * log(std^2))
        # eps = torch.randn_like(std)  # random ~ N(0, 1)
        # z = eps.mul(std).add_(mu)
        # z = self.fc_z(z)
        # z = z.view(-1, 64, 2, 2)
        #
        # rx = F.elu(self.t_conv1(z))
        # rx = F.elu(self.t_conv2(rx))
        # rx = F.elu(self.t_conv3(rx))
        # rx = torch.sigmoid(self.t_conv4(rx))

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


# def loss_fn(outputs, labels):
#     """
#     Compute the cross entropy loss given outputs and labels.
#
#     Args:
#         outputs: (Variable) dimension batch_size x 6 - output of the model
#         labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]
#
#     Returns:
#         loss (Variable): cross entropy loss for all images in the batch
#
#     Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
#           demonstrates how you can easily define a custom loss function.
#     """
#     num_examples = outputs.size()[0]
#     return -torch.sum(outputs[range(num_examples), labels])/num_examples

# loss_fn = nn.BCELoss()
#
# def mse(outputs, inputs):
#     """
#     Compute the MSE, given the output and input images.
#
#     Args:
#         outputs: (np.ndarray) dimension batch_size x image shape
#         inputs:  (np.ndarray) dimension batch_size x image shape
#
#     Returns: (float) root MSE
#     """
#     mse = torch.mean(torch.pow(torch.sub(outputs, inputs), 2))
#     return mse.cpu().detach().numpy()
#
#     # return np.sqrt(np.mean(np.subtract(outputs, inputs, dtype=float)**2))
#
#
# def accuracy(outputs, inputs):
#     """
#     Return 0, because I haven't implemented it yet
#
#     Args:
#         outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
#         labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]
#
#     Returns: (float) accuracy in [0,1]
#     """
#     # outputs = np.argmax(outputs, axis=1)
#     # return np.sum(outputs==labels)/float(labels.size)
#
#     return 0
#
#
# # maintain all metrics required in this dictionary- these are used in the training and evaluation loops
# metrics = {
#     # 'accuracy': accuracy,
#     'mse': mse,
#     # could add more metrics such as accuracy for each token type
# }
