import numpy as np
import imageio
import PIL
import os
import argparse
import matplotlib.pyplot as plt
import sys

import pdb
import pickle
import datetime

import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms

# sys.path.append('/scratch/cloned_repositories/torch-summary')
# from torchsummary import summary

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse

import model.vae as vae
import model.data_loader as data_loader
from model.data_loader import eval_transformer
import utils

torch.manual_seed(42)
torch.cuda.manual_seed(42)

def pickle_vars(query_stride, compare_stride, nr_similar_patches, vae_version, mses, ssims, psnrs):
    pickle_file_path = '/home/niaki/temp/20201214_VAE_experiments_zimnica/pickled_vars_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.pickle'
    try:
        pickle.dump((query_stride, compare_stride, nr_similar_patches, vae_version, mses, ssims, psnrs), open(pickle_file_path, "wb"))
    except Exception as e:
        print("Problem while trying to pickle: ", str(e))

# def calculate_psnr(img1, img2, max_value=1.0):
#     """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
#     mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
#     if mse == 0:
#         return 100
#     return 20 * np.log10(max_value / (np.sqrt(mse)))

# psnr = calculate_psnr

def compute_descriptor(descr, patch):
    if descr == model_vae:
        patch = patch.unsqueeze(0).unsqueeze(0)
        encoding, _, _ = descr.encode(patch)
        return encoding
    else:
        raise Exception("This descriptor not supported!")

def calculate_SSDs_for_descr(image, patch_size=65, query_stride=65, compare_stride=65, nr_similar_patches=6, eps=0.0001):

    assert len(list(image.shape)) == 2, "Image should be greyscale!"

    image_height = image.shape[0]
    image_width = image.shape[1]

    query_x_coords = []
    query_y_coords = []

    results_noisy_descr_patches_diffs = {}
    results_noisy_descr_patches_x_coords = {}
    results_noisy_descr_patches_y_coords = {}
    results_noisy_descr_patches_positions = {}

    counter_query_patches = 0

    # just for the sake of output
    total_nr_query_patches = len(range(0, image_width - patch_size + 1, query_stride)) * len(
        range(0, image_height - patch_size + 1, query_stride))

    for y_query in range(0, image_width - patch_size + 1, query_stride):
        for x_query in range(0, image_height - patch_size + 1, query_stride):
            sys.stdout.write("\r" + str(counter_query_patches + 1) + "/" + str(total_nr_query_patches))

            query_x_coords.append(x_query)
            query_y_coords.append(y_query)

            query_patch = image[x_query: x_query + patch_size, y_query: y_query + patch_size]

            query_patch_descr = compute_descriptor(model_vae, query_patch)

            counter_compare_patches = 0

            patches_diffs = [1000000000]
            patches_x_coords = [-1]
            patches_y_coords = [-1]
            patches_positions = [-1]

            for y_compare in range(0, image_width - patch_size + 1, compare_stride):
                for x_compare in range(0, image_height - patch_size + 1, compare_stride):

                    compare_patch = image[x_compare: x_compare + patch_size, y_compare: y_compare + patch_size]

                    compare_patch_descr = compute_descriptor(model_vae, compare_patch)

                    # diff = mse(query_patch_descr, compare_patch_descr)
                    diff = mse(query_patch_descr.detach().cpu().numpy(), compare_patch_descr.detach().cpu().numpy())

                    if diff < eps or (y_query == y_compare and x_query == x_compare):
                        counter_compare_patches += 1
                        continue

                    # sorting
                    for i in range(len(patches_diffs)):
                        if diff < patches_diffs[i]:
                            patches_diffs.insert(i, diff)
                            patches_x_coords.insert(i, x_compare)
                            patches_y_coords.insert(i, y_compare)
                            patches_positions.insert(i, counter_compare_patches)
                            break

                    counter_compare_patches += 1

            results_noisy_descr_patches_diffs[counter_query_patches] = patches_diffs[:nr_similar_patches]
            results_noisy_descr_patches_x_coords[counter_query_patches] = patches_x_coords[:nr_similar_patches]
            results_noisy_descr_patches_y_coords[counter_query_patches] = patches_y_coords[:nr_similar_patches]
            results_noisy_descr_patches_positions[counter_query_patches] = patches_positions[:nr_similar_patches]

            counter_query_patches += 1

    mses = []
    ssims = []
    psnrs = []

    for q_it in range(total_nr_query_patches):

        # getting the query patch
        x_query = query_x_coords[q_it]
        y_query = query_y_coords[q_it]
        query_patch = image[x_query: x_query + patch_size, y_query: y_query + patch_size]
        query_patch = query_patch.cpu().numpy()

        for c_it in range(nr_similar_patches):

            # getting the compare patch as found by looking at the closest encodings
            x_compare = results_noisy_descr_patches_x_coords[q_it][c_it]
            y_compare = results_noisy_descr_patches_y_coords[q_it][c_it]
            compare_patch = image[x_compare: x_compare + patch_size, y_compare: y_compare + patch_size]
            compare_patch = compare_patch.cpu().numpy()

            # calculating different difference between the query and the compare patch
            actual_diff = mse(query_patch, compare_patch)
            mses.append(actual_diff)

            dr_max = max(query_patch.max(), compare_patch.max())
            dr_min = min(query_patch.min(), compare_patch.min())

            diff_ssim = ssim(query_patch, compare_patch, data_range=dr_max - dr_min)
            ssims.append(diff_ssim)
            diff_psnr = psnr(query_patch, compare_patch)#, data_range=dr_max - dr_min)
            psnrs.append(diff_psnr)

    mses = np.array(mses)
    ssims = np.array(ssims)
    psnrs = np.array(psnrs)

    return mses, ssims, psnrs


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


patch_size = 64


vae_version = 'vae_20201213_102539'

args = parser.parse_args()
weights_path = os.path.join(args.weights_dir, vae_version, 'best.pth.tar')
# data_dir = '/scratch/image_datasets/3_65x65/ready/'
json_path = os.path.join(args.model_dir, 'params.json')

model_vae = vae.BetaVAE(128)
model_vae.load_state_dict(torch.load(weights_path)['state_dict'])
model_vae.eval()
model_vae = model_vae.cuda()

# summary(model_vae, (1, 64, 64))

# params = utils.Params(json_path)
# params.cuda = torch.cuda.is_available()  # use GPU if available
# dataloaders = data_loader.fetch_dataloader(['test'], args.data_dir, params)
# test_dl = dataloaders['test']

image = PIL.Image.open('/home/niaki/Downloads/ViandenCastle_autumm-1-von-1-scaled_gray.jpg')

image = transforms.ToTensor()((image)).cuda()
image = image[0]

query_stride = patch_size * 8
compare_stride = patch_size
nr_similar_patches = 6
mses, ssims, psnrs = calculate_SSDs_for_descr(image, patch_size=patch_size, query_stride=query_stride, compare_stride=compare_stride, nr_similar_patches=nr_similar_patches, eps=0.0001)
print()
print('MSEs', np.mean(mses))
print('SSIMs', np.mean(ssims))
print('PSNRs', np.mean(psnrs))

pickle_vars(query_stride, compare_stride, nr_similar_patches, vae_version, mses, ssims, psnrs)
with open('/home/niaki/temp/20201214_VAE_experiments_zimnica/' + vae_version + '.txt', 'a') as the_file:
    the_file.write(str(query_stride) + ' ' + str(compare_stride) + ' ' + str(nr_similar_patches) + '\n')
    the_file.write('MSEs  ' + str(np.mean(mses)) + '\n')
    the_file.write('SSIMs ' + str(np.mean(ssims)) + '\n')
    the_file.write('PSNRs  ' + str(np.mean(psnrs)) + '\n')
# pdb.set_trace()