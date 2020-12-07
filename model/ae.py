import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import pdb

# define the NN architecture
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        self.zeropad1 = nn.ZeroPad2d(1)
        # conv layer (depth from 1 --> 32), 3x3 kernels
        self.conv1 = nn.Conv2d(1, 32, 3, padding=0)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool1 = nn.MaxPool2d(2, 2)
        self.zeropad2 = nn.ZeroPad2d(1)
        # conv layer (depth from 32 --> 32), 3x3 kernels
        self.conv2 = nn.Conv2d(32, 32, 3, padding=0)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool2 = nn.MaxPool2d(2, 2)
        self.zeropad3 = nn.ZeroPad2d(1)
        # conv layer (depth from 32 --> 32), 3x3 kernels
        self.conv3 = nn.Conv2d(32, 32, 3, padding=0)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool3 = nn.MaxPool2d(2, 2)

        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(32, 32, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(32, 32, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(32, 1, 2, stride=2)

    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.elu(self.conv1(self.zeropad1(x)))
        x = self.pool1(x)
        # add second hidden layer
        x = F.elu(self.conv2(self.zeropad2(x)))
        x = self.pool2(x)
        # add third hidden layer
        x = F.elu(self.conv3(self.zeropad3(x)))
        x = self.pool3(x)  # compressed representation

        # pdb.set_trace()

        ## decode ##
        # add transpose conv layers, with eelu activation function
        x = F.elu(self.t_conv1(x))
        # add transpose conv layers, with eelu activation function
        x = F.elu(self.t_conv2(x))
        # output layer (with sigmoid for scaling from 0 to 1)
        x = torch.sigmoid(self.t_conv3(x))

        return x


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

loss_fn = nn.BCELoss()

def mse(outputs, inputs):
    """
    Compute the MSE, given the output and input images.

    Args:
        outputs: (np.ndarray) dimension batch_size x image shape
        inputs:  (np.ndarray) dimension batch_size x image shape

    Returns: (float) root MSE
    """
    mse = torch.mean(torch.pow(torch.sub(outputs, inputs), 2))
    return mse.cpu().detach().numpy()

    # return np.sqrt(np.mean(np.subtract(outputs, inputs, dtype=float)**2))


def accuracy(outputs, inputs):
    """
    Return 0, because I haven't implemented it yet

    Args:
        outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns: (float) accuracy in [0,1]
    """
    # outputs = np.argmax(outputs, axis=1)
    # return np.sum(outputs==labels)/float(labels.size)

    return 0


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    # 'accuracy': accuracy,
    'mse': mse,
    # could add more metrics such as accuracy for each token type
}


# FROM KAGGLE TUTORIAL
#
#
# # initialize the NN
# model = ConvAutoencoder()
# print(model)
#
# criterion = nn.BCELoss()
#
# # specify loss function
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
#
# # number of epochs to train the model
# n_epochs = 100
#
# for epoch in range(1, n_epochs + 1):
#     # monitor training loss
#     train_loss = 0.0
#
#     ###################
#     # train the model #
#     ###################
#     for data in train_loader:
#         # _ stands in for labels, here
#         # no need to flatten images
#         images, _ = data
#         # clear the gradients of all optimized variables
#         optimizer.zero_grad()
#         # forward pass: compute predicted outputs by passing inputs to the model
#         outputs = model(images)
#         # calculate the loss
#         loss = criterion(outputs, images)
#         # backward pass: compute gradient of the loss with respect to model parameters
#         loss.backward()
#         # perform a single optimization step (parameter update)
#         optimizer.step()
#         # update running training loss
#         train_loss += loss.item() * images.size(0)
#
#     # print avg training statistics
#     train_loss = train_loss / len(train_loader)
#     print('Epoch: {} \tTraining Loss: {:.6f}'.format(
#         epoch,
#         train_loss
#     ))
