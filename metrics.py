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


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'mse': mse,
}
