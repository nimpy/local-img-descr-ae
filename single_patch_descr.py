import numpy as np
import torch

import model.ae as ae
import model.vae as vae


def encode_single_patch(model, patch):  # TODO ensure the types of model and patch are correct
    model.eval()
    variational = isinstance(model, vae.BetaVAE)
    if variational:
        encoding, _, _ = model.encode(patch)
    else:
        encoding = model.encode(patch)
    return encoding


if __name__ == '__main__':
    weights_path = '/scratch/image_datasets/3_65x65/ready/weights/vae_20201212_100238/best.pth.tar'

    model = vae.BetaVAE(128)  # ae.AE()
    model.load_state_dict(torch.load(weights_path)['state_dict'])

    patch = np.random.random((1, 1, 64, 64))
    patch = torch.from_numpy(patch).float()
    encoding = encode_single_patch(model, patch)
    print(encoding)


