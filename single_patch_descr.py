import numpy as np
import torch

import models.ae as ae
import models.vae as vae
import models.ae_ir as ae_ir
import models.vae_ir as vae_ir


def encode_single_patch(model, patch):  # TODO ensure the types of model and patch are correct
    model.eval()
    variational = isinstance(model, vae.BetaVAE) or isinstance(model, vae_ir.BetaVAE)
    if variational:
        encoding, _, _ = model.encode(patch)
    else:
        encoding = model.encode(patch)
    return encoding


if __name__ == '__main__':
    weights_path = 'weights_pub/vae_best.pth.tar'  # or path to another model, e.g. 'weights_pub/ae_ir_best.pth.tar'

    model = vae.BetaVAE(128)  # or another model, e.g. ae.AE()
    model.load_state_dict(torch.load(weights_path)['state_dict'])

    patch = np.random.random((1, 1, 64, 64))  # or an actual image patch
    patch = torch.from_numpy(patch).float()
    encoding = encode_single_patch(model, patch)
    print(encoding)


