# from PIL import Image
import os
import argparse
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse

import torch
from torch.autograd import Variable

import data_loader as data_loader
from models.vae import BetaVAE
import utilities

torch.manual_seed(42)
torch.cuda.manual_seed(42)

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
weights_path = os.path.join(args.weights_dir, 'vae_20201211_151545/best.pth.tar')
json_path = os.path.join(args.model_dir, 'params.json')

model = BetaVAE(128)#AE()
model.load_state_dict(torch.load(weights_path)['state_dict'])
model.eval()

params = utilities.Params(json_path)
params.cuda = torch.cuda.is_available()

if params.cuda:
    model = model.cuda()
dataloaders = data_loader.fetch_dataloader(['test'], args.data_dir, params)
test_dl = dataloaders['test']

counter = 0
counter_vis = 0
diff_mse_cum = 0
diff_ssim_cum = 0
diff_psnr_cum = 0

for data_batch in test_dl:

    if params.cuda:
        data_batch = data_batch.cuda(non_blocking=True)

    data_batch = Variable(data_batch)

    output_batch, _, _ = model(data_batch)

    data_batch = data_batch.cpu().numpy()
    plt.imshow(data_batch[0][0], cmap='gray')
    plt.show()

    output_batch = output_batch.detach().cpu().numpy()
    plt.imshow(output_batch[0][0], cmap='gray')
    plt.show()

    counter += data_batch.shape[0]
    counter_vis += 1
    if counter_vis > 10:
        break

    for i in range(output_batch.shape[0]):
        diff_mse = mse(data_batch[i], output_batch[i])
        diff_mse_cum += diff_mse

        dr_max = max(data_batch[i].max(), output_batch[i].max())
        dr_min = min(data_batch[i].min(), output_batch[i].min())

        diff_ssim = ssim(data_batch[i,0], output_batch[i,0], data_range=dr_max - dr_min)
        diff_ssim_cum += diff_ssim

        diff_psnr = psnr(data_batch[i,0], output_batch[i,0], data_range=dr_max - dr_min)
        diff_psnr_cum += diff_psnr

diff_mse_average = diff_mse_cum / counter
diff_ssim_average = diff_ssim_cum / counter
diff_psnr_average = diff_psnr_cum / counter
print(diff_mse_average)
print(diff_ssim_average)
print(diff_psnr_average)
# print("MSEs", diff_mse_average)
# print("SSIMs", diff_ssim_average)
# print("PSNRs", diff_psnr_average)



