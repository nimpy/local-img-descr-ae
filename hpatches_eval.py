import os
import cv2
import numpy as np
import datetime

import torch

import wandb

import dill
import json
from tabulate import tabulate as tb

import models.ae as ae
import models.vae as vae
from single_patch_descr import encode_single_patch
from utilities import default_to_regular_dict

import sys
# sys.path.append('/scratch/cloned_repositories/hpatches-benchmark/python')
# from extract_opencv_sift import hpatches_sequence
sys.path.append('/scratch/cloned_repositories/hpatches-benchmark/python')
from utils.tasks import methods, eval_verification, eval_matching, eval_retrieval
from utils.hpatch import load_descrs

hpatches_types = ['ref','e1','e2','e3','e4','e5','h1','h2','h3','h4','h5','t1','t2','t3','t4','t5']

with open(os.path.join("/scratch/cloned_repositories/hpatches-benchmark/python/utils", "splits.json")) as f:
    splits = json.load(f)
split_c = splits['c']
hpatches_seqs = split_c['test']

hpatches_data_dir = "/scratch/hpatches/hpatches-benchmark/data/hpatches-release"
hpatches_seqs = [os.path.join(hpatches_data_dir, test_seq) for test_seq in hpatches_seqs]

encodings_base_dir = "/scratch/cloned_repositories/hpatches-benchmark/data"
encodings_dir = os.path.join(encodings_base_dir, 'ae1')#datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

results_dir = "/scratch/cloned_repositories/hpatches-benchmark/results"

ft = {'e':'Easy','h':'Hard','t':'Tough'}  # TODO: rename


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

    hpatches_collect_results(use_wandb)

    return 0


def hpatches_extract_descrs(model):  # TODO: speed up!
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


def hpatches_collect_results(use_wandb):
    descr = 'ae_bak'
    result_verification = results_verification(descr, split_c)
    print()
    result_matching = results_matching(descr, split_c)
    print()
    result_retrieval = results_retrieval(descr, split_c)
    print()

    # results = {}
    # results['verification'] = result_verification
    # results['matching'] = result_matching
    # results['retrieval'] = result_retrieval
    # print(results)
    print(result_verification)
    print(result_matching)
    print(result_retrieval)

    if use_wandb:
        wandb.log({"verification": result_verification, "matching": result_matching, "retrieval": result_retrieval})


def results_verification(desc, splt):
    v = {'balanced':'auc','imbalanced':'ap'}
    inter_intra = ['inter', 'intra']
    res = dill.load(open(os.path.join(results_dir, desc+"_verification_"+splt['name']+".p"), "rb"))
    for r in v:
        print("%s - %s variant (%s) " % (desc.upper(),r.capitalize(),v[r]))
        heads = ["Noise","Inter","Intra"]
        results = []
        for t in ['e','h','t']:
            results.append([ft[t], res[t]['inter'][r][v[r]],res[t]['intra'][r][v[r]]])
        print(tb(results,headers=heads))
        print()

    # logging
    for diff_level in ft.keys():
        for ii in inter_intra:
            for version in v.keys():
                if version == 'balanced':
                    res[diff_level][ii][version].pop('fpr')
                    res[diff_level][ii][version].pop('tpr')
                else:
                    res[diff_level][ii][version].pop('pr')
                    res[diff_level][ii][version].pop('rc')

    return default_to_regular_dict(res)


def results_matching(desc, splt):
    res = dill.load(open(os.path.join(results_dir, desc+"_matching_"+splt['name']+".p"), "rb"))
    mAP = {'e':0,'h':0,'t':0}
    k_mAP = 0
    heads = [ft['e'],ft['h'],ft['t'],'mean']
    for seq in res:
        for t in ['e','h','t']:
            for idx in range(1,6):
                mAP[t] += res[seq][t][idx]['ap']
                k_mAP+=1
    k_mAP = k_mAP / 3.0
    print("%s - mAP " % (desc.upper()))

    results = [mAP['e']/k_mAP,mAP['h']/k_mAP,mAP['t']/k_mAP]
    results.append(sum(results)/float(len(results)))
    print(tb([results],headers=heads))
    print("\n")

    # logging
    return_results = {}
    for i, t in enumerate(['e', 'h', 't', 'mean']):
        return_results[t] = results[i]
    return return_results


def results_retrieval(desc, splt):
    res = dill.load(open(os.path.join(results_dir, desc+"_retrieval_"+splt['name']+".p"), "rb"))
    print("%s - mAP 10K queries " % (desc.upper()))
    n_q= float(len(res.keys()))
    heads = ['']
    mAP = dict.fromkeys(['e','h','t'])
    for k in mAP:
        mAP[k] = dict.fromkeys(res[0][k])
        for psize in mAP[k]:
            mAP[k][psize] = 0

    for qid in res:
        for k in mAP:
            for psize in mAP[k]:
                mAP[k][psize] +=  res[qid][k][psize]['ap']

    results = []
    for k in ['e','h','t']:
        heads = ['Noise']+sorted(mAP[k])
        r = []
        for psize in sorted(mAP[k]):
            r.append(mAP[k][psize]/n_q)
        results.append([ft[k]]+r)

    res = np.array(results)[:,1:].astype(np.float32)
    results.append(['mean']+list(np.mean(res,axis=0)))
    print(tb(results,headers=heads))

    # logging
    for k in ['e', 'h', 't']:
        for psize in sorted(mAP[k]):
            mAP[k][psize] /= n_q
    res_mean = list(np.mean(res,axis=0))
    mAP['mean'] = {}
    for i, psize in enumerate(sorted(mAP[k])):
        mAP['mean'][psize] = res_mean[i]
    return mAP


if __name__ == '__main__':

    weights_path = '/scratch/image_datasets/3_65x65/ready/weights/vae_20201212_100238/best.pth.tar'

    model = vae.BetaVAE(128)  # ae.AE()
    model.load_state_dict(torch.load(weights_path)['state_dict'])

    # TODO: delete the previous directory with descriptor's encodings, and make a new one

    use_wandb = False
    if use_wandb:
        wandb.login()
        wandb_run = wandb.init(project="temp")
        wandb.watch(model)

    # for i in range(10):
    hpatches_benchmark(model, use_wandb)

    if use_wandb:
        wandb_run.finish()


# TODO add retrieval.e.mean (mean for 100, 200, ...)
# TODO also add mean of every metric