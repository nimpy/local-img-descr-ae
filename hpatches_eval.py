import sys
import glob
import os
import cv2
import numpy as np
import datetime

import torch

import dill
import json


import models.ae as ae
import models.vae as vae
from single_patch_descr import encode_single_patch

import sys
# sys.path.append('/scratch/cloned_repositories/hpatches-benchmark/python')
# from extract_opencv_sift import hpatches_sequence
sys.path.append('/scratch/cloned_repositories/hpatches-benchmark/python')
from utils.tasks import methods, eval_verification, eval_matching, eval_retrieval
from utils.hpatch import load_descrs
from utils.results import results_methods

hpatches_types = ['ref','e1','e2','e3','e4','e5','h1','h2','h3','h4','h5','t1','t2','t3','t4','t5']
# hpatches_seqs = ["i_ski", "i_table", "i_troulos", "i_melon", "i_tools", "i_kions", "i_londonbridge", "i_nijmegen", "i_boutique", "i_parking", "i_steps", "i_fog", "i_leuven", "i_dc", "i_partyfood", "i_pool", "i_castle", "i_bologna", "i_smurf", "i_crownnight", "v_azzola", "v_tempera", "v_machines", "v_coffeehouse", "v_graffiti", "v_artisans", "v_maskedman", "v_talent", "v_bees", "v_dirtywall", "v_blueprint", "v_war", "v_adam", "v_pomegranate", "v_busstop", "v_weapons", "v_gardens", "v_feast", "v_man", "v_wounded"]  # c test

with open(os.path.join("/scratch/cloned_repositories/hpatches-benchmark/python/utils", "splits.json")) as f:
    splits = json.load(f)
split_c = splits['c']
hpatches_seqs = split_c['test']
# print(hpatches_seqs)

hpatches_data_dir = "/scratch/hpatches/hpatches-benchmark/data/hpatches-release"
hpatches_seqs = [os.path.join(hpatches_data_dir, test_seq) for test_seq in hpatches_seqs]

encodings_base_dir = "/scratch/cloned_repositories/hpatches-benchmark/data"
encodings_dir = os.path.join(encodings_base_dir, 'ae1')#datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

results_dir = "/scratch/cloned_repositories/hpatches-benchmark/results"


class hpatches_sequence:  # copied from HPatches repo (hence the non-standard case)
    """Class for loading an HPatches sequence from a sequence folder"""
    itr = hpatches_types
    def __init__(self,base):
        name = base.split('/')
        self.name = name[-1]
        self.base = base
        for t in self.itr:
            im_path = os.path.join(base, t+'.png')
            im = cv2.imread(im_path,0)
            self.N = im.shape[0]/65
            setattr(self, t, np.split(im, self.N))


def hpatches_benchmark(model, use_wandb):

    # hpatches_extract_descrs(model)

    # hpatches_eval_on_all_tasks()

    hpatches_collect_results()

    return 0

def hpatches_extract_descrs(model):
    # model.eval()
    # variational = isinstance(model, vae.BetaVAE)

    for seq_path in hpatches_seqs:
        seq = hpatches_sequence(seq_path)
        path = os.path.join(encodings_dir, seq.name)
        if not os.path.exists(path):
            os.makedirs(path)

        encoding_size = 128  # TODO don't leave this hard-coded
        encodings = np.zeros((int(seq.N), encoding_size))

        print(seq.name)
        for type in hpatches_types:
            for i, patch in enumerate(getattr(seq, type)):
                # mi = np.mean(patch)    # trivial (mi,sigma) descriptor
                # sigma = np.std(patch)  # trivial (mi,sigma) descriptor
                # encodings[i] = np.array([mi,sigma])
                patch = patch / 255.0
                patch = np.expand_dims(patch, axis=0)
                patch = np.expand_dims(patch, axis=0)
                patch = torch.from_numpy(patch).float()
                encodings[i] = encode_single_patch(model, patch).detach().numpy()
            np.savetxt(os.path.join(path, type + '.csv'), encodings, delimiter=',')  # X is an array

    return


def hpatches_eval_on_all_tasks():

    descr = load_descrs("/scratch/cloned_repositories/hpatches-benchmark/data/ae_bak")
    for method_name in methods.keys():
        print(method_name)
        results_path = os.path.join(results_dir, "ae_bak" + "_" + method_name + "_" + split_c['name'] + ".p")
        print(results_path)

        res = methods[method_name](descr, split_c)
        dill.dump(res, open(results_path, "wb"))


def hpatches_collect_results():
    for results_method_name in results_methods.keys():
        print("%s task results:" % (results_method_name.capitalize()))
        descr = 'ae_bak'
        results_methods[results_method_name](descr, split_c)
        print()



if __name__ == '__main__':

    weights_path = '/scratch/image_datasets/3_65x65/ready/weights/vae_20201212_100238/best.pth.tar'

    model = vae.BetaVAE(128)  # ae.AE()
    model.load_state_dict(torch.load(weights_path)['state_dict'])

    # TODO: delete the previous directory with descriptor's encodings, and make a new one

    use_wandb = False
    hpatches_benchmark(model, use_wandb)
