import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import pdb

class BetaVAE(nn.Module):

    def __init__(self, latent_size=32, beta=1):
        super(BetaVAE, self).__init__()

        self.latent_size = latent_size
        self.beta = beta

        # encoder
        self.encoder = nn.Sequential(
            self._conv(1, 32),
            self._conv(32, 32),
            self._conv(32, 64),
            self._conv(64, 64),
        )
        self.fc_mu = nn.Linear(256, latent_size)
        self.fc_var = nn.Linear(256, latent_size)

        # decoder
        self.decoder = nn.Sequential(
            self._deconv(64, 64),
            self._deconv(64, 32),
            self._deconv(32, 32, 1),
            self._deconv(32, 1),
            nn.Sigmoid()
        )
        self.fc_z = nn.Linear(latent_size, 256)

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(-1, 256)
        return self.fc_mu(x), self.fc_var(x)

    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)  # e^(1/2 * log(std^2))
        eps = torch.randn_like(std)  # random ~ N(0, 1)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        z = self.fc_z(z)
        z = z.view(-1, 64, 2, 2)
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        rx = self.decode(z)
        return rx, mu, logvar

    def _conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=4, stride=2
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    # out_padding is used to ensure output size matches EXACTLY of conv2d;
    # it does not actually add zero-padding to output :)
    def _deconv(self, in_channels, out_channels, out_padding=0):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=4, stride=2, output_padding=out_padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

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

    # def save_model(self, file_path, num_to_keep=1):
    #     utils.save(self, file_path, num_to_keep)
    #
    # def load_model(self, file_path):
    #     utils.restore(self, file_path)
    #
    # def load_last_model(self, dir_path):
    #     return utils.restore_latest(self, dir_path)








#
# # define the NN architecture
# class VariationalAutoencoder(nn.Module):
#     def __init__(self, latent_size=128, gamma=1):
#         super(VariationalAutoencoder, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3, padding=0)  # conv layer (depth from 1 --> 32), 3x3 kernels
#         self.conv2 = nn.Conv2d(32, 32, 3, padding=0)  # conv layer (depth from 32 --> 64), 3x3 kernels
#         self.conv3 = nn.Conv2d(32, 64, 3, padding=0)  # conv layer (depth from 32 --> 64), 3x3 kernels
#         self.conv4 = nn.Conv2d(64, 64, 3, padding=0)  # conv layer (depth from 32 --> 32), 3x3 kernels
#
#         self.fc_mu = nn.Linear(256, latent_size)
#         self.fc_var = nn.Linear(256, latent_size)
#
#         self.t_conv1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
#         self.t_conv2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
#         self.t_conv2 = nn.ConvTranspose2d(32, 32, 2, stride=2)
#         self.t_conv3 = nn.ConvTranspose2d(32, 1, 2, stride=2)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(self.zeropad1(x)))
#         x = self.pool1(x)
#
#         x = F.elu(self.conv2(self.zeropad2(x)))
#         x = self.pool2(x)
#
#         x = F.elu(self.conv3(self.zeropad3(x)))
#         x = self.pool3(x)  # compressed representation
#
#
#         ## decode ##
#         # add transpose conv layers, with eelu activation function
#         x = F.elu(self.t_conv1(x))
#         # add transpose conv layers, with eelu activation function
#         x = F.elu(self.t_conv2(x))
#         # output layer (with sigmoid for scaling from 0 to 1)
#         x = torch.sigmoid(self.t_conv3(x))
#
#         return x
#
#
# # def loss_fn(outputs, labels):
# #     """
# #     Compute the cross entropy loss given outputs and labels.
# #
# #     Args:
# #         outputs: (Variable) dimension batch_size x 6 - output of the model
# #         labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]
# #
# #     Returns:
# #         loss (Variable): cross entropy loss for all images in the batch
# #
# #     Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
# #           demonstrates how you can easily define a custom loss function.
# #     """
# #     num_examples = outputs.size()[0]
# #     return -torch.sum(outputs[range(num_examples), labels])/num_examples
#
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
