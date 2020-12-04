# from PIL import Image
import numpy as np
import imageio
import os
import argparse
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

import model.data_loader as data_loader
from model.ae import ConvAutoencoder
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/scratch/image_datasets/3_65x65/ready',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='model/',
                    help="Directory containing params.json")
parser.add_argument('--weights_dir', default='/scratch/image_datasets/3_65x65/ready/weights',
                    help="Directory where weights will be saved")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'


args = parser.parse_args()
weights_path = os.path.join(args.weights_dir, 'ae_20201204_145628/last.pth.tar')
# data_dir = '/scratch/image_datasets/3_65x65/ready/'
json_path = os.path.join(args.model_dir, 'params.json')

model = ConvAutoencoder()
model.load_state_dict(torch.load(weights_path)['state_dict'])
model.eval()

model = model.cuda()

# dl = DataLoader(data_loader.PatchesDataset('/scratch/image_datasets/3_65x65/ready/test/class0/', data_loader.eval_transformer), batch_size=1, shuffle=False)

params = utils.Params(json_path)
# use GPU if available
params.cuda = torch.cuda.is_available()

dataloaders = data_loader.fetch_dataloader(['test'], args.data_dir, params)
test_dl = dataloaders['test']

# patch1 = imageio.imread('/scratch/image_datasets/3_65x65/ready/test/class0/patch_65x65_000016.bmp')
# patch2 = imageio.imread('/scratch/image_datasets/3_65x65/ready/test/class0/patch_65x65_000017.bmp')  # 26
# patches = np.array([patch1, patch2])
# patches = data_loader.eval_transformer(patches)
#
# print(model(patches))

counter = 0

for data_batch in test_dl:

    # move to GPU if available
    data_batch = data_batch.cuda(non_blocking=True)
    # fetch the next evaluation batch
    data_batch = Variable(data_batch)

    # compute model output
    output_batch = model(data_batch)

    data_batch = data_batch.cpu().numpy()
    plt.imshow(data_batch[0][0], cmap='gray')
    plt.show()

    output_batch = output_batch.detach().cpu().numpy()
    plt.imshow(output_batch[0][0], cmap='gray')
    plt.show()

    counter += 1
    if counter > 10:
        break