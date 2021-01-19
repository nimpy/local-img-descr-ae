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
from utilities import default_to_regular_dict, pretty_dict

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

    hpatches_extract_descrs(model)

    # hpatches_eval_on_all_tasks()

    # hpatches_collect_results(use_wandb)

    return 0


def hpatches_extract_descrs(model):
    model.eval()
    variational = isinstance(model, vae.BetaVAE)

    for seq_path in hpatches_seqs:
        seq = hpatches_sequence(seq_path)
        path = os.path.join(encodings_dir, seq.name)
        if not os.path.exists(path):
            os.makedirs(path)

        print(seq.name)
        for type in hpatches_types:
            batch = getattr(seq, type)
            batch = np.array(batch)
            batch = batch / 255.0
            batch = np.expand_dims(batch, axis=1)
            batch = torch.from_numpy(batch).float()
            if variational:
                batch_encodings, _, _ = model.encode(batch)
            else:
                batch_encodings = model.encode(batch)
            batch_encodings = batch_encodings.detach().numpy()

            # reshape to (batch_size, flattened_encoding)
            batch_encodings = batch_encodings.reshape(batch_encodings.shape[0], np.product(batch_encodings.shape[1:]))
            np.savetxt(os.path.join(path, type + '.csv'), batch_encodings, delimiter=',')

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
    print()
    print(result_verification)
    print(result_matching)
    print(result_retrieval)
    print('\nVERIFICATION DICT')
    pretty_dict(result_verification)
    print('\nMATCHING DICT')
    pretty_dict(result_matching)
    print('\nRETRIEVAL DICT')
    pretty_dict(result_retrieval)
    print()
    mean_verification = result_verification['mean']['mean']
    mean_matching = result_matching['mean']
    mean_retrieval = result_retrieval['mean']['mean']
    overall_mean = np.mean(np.array([mean_verification, mean_matching, mean_retrieval]))
    print("OVERALL MEAN:", overall_mean)

    if use_wandb:
        wandb.log({"verification": result_verification, "matching": result_matching, "retrieval": result_retrieval, "overall_mean": overall_mean})


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

    res = default_to_regular_dict(res)
    res['mean'] = {}
    for ii in inter_intra:
        res['mean'][ii] = {}
        mean_ii_cum = 0
        for version in v.keys():
            res['mean'][ii][version] = {}
            res['mean'][ii][version][v[version]] = (res['e'][ii][version][v[version]] +
                                                    res['h'][ii][version][v[version]] +
                                                    res['t'][ii][version][v[version]]) / 3.0
            mean_ii_cum += res['mean'][ii][version][v[version]]
        res['mean'][ii]['mean'] = mean_ii_cum / 2.0
    res['mean']['mean'] = (res['mean']['inter']['mean'] + res['mean']['intra']['mean']) / 2.0
    return res


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
    for i, k in enumerate(['e', 'h', 't']):
        for psize in sorted(mAP[k]):
            mAP[k][psize] /= n_q
        mAP[k]['mean'] = np.mean(res, axis=1)[i]
    res_mean = list(np.mean(res,axis=0))
    mAP['mean'] = {}
    for i, psize in enumerate([100, 500, 1000, 5000, 10000, 15000, 20000]):
        mAP['mean'][psize] = res_mean[i]
    mAP['mean']['mean'] = res.mean()
    return mAP


if __name__ == '__main__':

    weights_path = '/scratch/image_datasets/3_65x65/ready/weights/ae_20201207_143916/best.pth.tar' # vae_20201212_100238

    model = ae.AE()  # vae.BetaVAE(128)  # ae.AE()
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

