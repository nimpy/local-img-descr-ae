import os
import cv2
import numpy as np
import datetime
import logging

import torch
import wandb

import dill
import json
from tabulate import tabulate as tb

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse

from torch.autograd import Variable

import models.ae as ae
import models.vae as vae
import models.ae_ir as ae_ir
import models.vae_ir as vae_ir

from utilities import default_to_regular_dict, pretty_dict
import utilities
import data_loader

import sys
sys.path.append('/scratch/cloned_repositories/hpatches-benchmark/python')
from utils.tasks import methods, eval_verification, eval_matching, eval_retrieval
from utils.hpatch import load_descrs


hpatches_types = ['ref','e1','e2','e3','e4','e5','h1','h2','h3','h4','h5','t1','t2','t3','t4','t5']
ft = {'e': 'Easy', 'h': 'Hard', 't': 'Tough'}

# ugly local paths...
with open(os.path.join("/scratch/cloned_repositories/hpatches-benchmark/python/utils", "splits.json")) as f:
    splits = json.load(f)
split_c = splits['c']
hpatches_seqs = split_c['test']

hpatches_data_dir = "/scratch/hpatches/hpatches-benchmark/data/hpatches-release"
hpatches_seqs = [os.path.join(hpatches_data_dir, test_seq) for test_seq in hpatches_seqs]

encodings_base_dir = "/scratch/cloned_repositories/hpatches-benchmark/data"

results_dir = "/scratch/cloned_repositories/hpatches-benchmark/results"


# taken and adjusted from:
# repository https://github.com/hpatches/hpatches-benchmark
# file hpatches-benchmark/python/extract_opencv_sift.py
class HPatchesSequence:
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


def hpatches_benchmark_a_model(model, model_version, use_wandb):

    logging.info("===== Calculating MSE, SSIM, and PSNR metrics =====")
    mse, ssim, psnr = calculate_approximate_evaluation_metrics_on_test_set(model)
    logging.info("... they are, respectively: " + str(mse) + ', ' + str(ssim) + ', ' + str(psnr))

    logging.info("===== Extracting the encodings using the descriptor =====")
    hpatches_extract_encodings(model, model_version)

    logging.info("\n===== Evaluating the descriptor on all tasks =====")
    hpatches_eval_on_all_tasks(model_version)

    logging.info("\n===== Collecting the results from tasks =====")
    result_verification, result_matching, result_retrieval, overall_mean = hpatches_collect_results(model_version)

    if use_wandb:
        wandb.log({"verification": result_verification, "matching": result_matching, "retrieval": result_retrieval,
                   "hpatches_overall": overall_mean, "mse": mse, "ssim": ssim, "psnr": psnr})

    return overall_mean


# some code copied from fast_approximate_evaluation.py
def calculate_approximate_evaluation_metrics_on_test_set(model):

    params = utilities.Params('models/params.json')
    params.cuda = torch.cuda.is_available()

    variational = isinstance(model, vae.BetaVAE) or isinstance(model, vae_ir.BetaVAE)

    dataloaders = data_loader.fetch_dataloader(['test'], '/scratch/image_datasets/3_65x65/ready', params, batch_size=32)
    test_dl = dataloaders['test']

    counter = 0
    diff_mse_cum = 0
    diff_ssim_cum = 0
    diff_psnr_cum = 0

    model.eval()

    for data_batch in test_dl:

        data_batch = data_batch.cuda(non_blocking=True)
        data_batch = Variable(data_batch)

        if variational:
            output_batch, _, _ = model(data_batch)
        else:
            output_batch = model(data_batch)

        data_batch = data_batch.cpu().numpy()
        output_batch = output_batch.detach().cpu().numpy()

        counter += data_batch.shape[0]

        for i in range(output_batch.shape[0]):
            diff_mse = mse(data_batch[i], output_batch[i])
            diff_mse_cum += diff_mse

            dr_max = max(data_batch[i].max(), output_batch[i].max())
            dr_min = min(data_batch[i].min(), output_batch[i].min())

            diff_ssim = ssim(data_batch[i, 0], output_batch[i, 0], data_range=dr_max - dr_min)
            diff_ssim_cum += diff_ssim

            diff_psnr = psnr(data_batch[i, 0], output_batch[i, 0], data_range=dr_max - dr_min)
            diff_psnr_cum += diff_psnr

    diff_mse_average = diff_mse_cum / counter
    diff_ssim_average = diff_ssim_cum / counter
    diff_psnr_average = diff_psnr_cum / counter

    return diff_mse_average, diff_ssim_average, diff_psnr_average


def hpatches_extract_encodings(model, model_version):
    encodings_dir = os.path.join(encodings_base_dir, model_version)

    model = model.cpu()

    model.eval()
    variational = isinstance(model, vae.BetaVAE) or isinstance(model, vae_ir.BetaVAE)

    for seq_path in hpatches_seqs:
        seq = HPatchesSequence(seq_path)
        path = os.path.join(encodings_dir, seq.name)
        logging.info(seq.name)

        if not os.path.exists(path):
            os.makedirs(path)
        elif len(os.listdir(path)) == len(hpatches_types):
            logging.info('The encodings already exist! Not gonna calculate them again!')
            continue
        else:
            logging.info('Might override some previously calculated encodings!')

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


def hpatches_eval_on_all_tasks(model_version):

    descr = load_descrs(os.path.join(encodings_base_dir, model_version))
    for method_name in methods.keys():
        results_path = os.path.join(results_dir, model_version + "_" + method_name + "_" + split_c['name'] + ".p")
        logging.info("Evaluating HPatches task: " + method_name)
        logging.info("The results will be saved under:\n    " + results_path)

        res = methods[method_name](descr, split_c)
        dill.dump(res, open(results_path, "wb"))


def hpatches_collect_results(model_version):
    result_verification = results_verification(model_version, split_c)
    logging.info("\n")
    result_matching = results_matching(model_version, split_c)
    logging.info("\n")
    result_retrieval = results_retrieval(model_version, split_c)

    logging.info("\nDicts:")
    logging.info(result_verification)
    logging.info(result_matching)
    logging.info(result_retrieval)
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
    logging.info("OVERALL MEAN: " +  str(overall_mean))

    return result_verification, result_matching, result_retrieval, overall_mean


# taken and adjusted from:
# repository https://github.com/hpatches/hpatches-benchmark
# file hpatches-benchmark/python/utils/results.py
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


# taken and adjusted from:
# repository https://github.com/hpatches/hpatches-benchmark
# file hpatches-benchmark/python/utils/results.py
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


# taken and adjusted from:
# repository https://github.com/hpatches/hpatches-benchmark
# file hpatches-benchmark/python/utils/results.py
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



