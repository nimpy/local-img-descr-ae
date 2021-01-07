import torch

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
