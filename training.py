"""Train the model"""

import argparse
import logging
import os

import numpy as np
import math
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import datetime
from pathlib import Path

import sys
sys.path.append('/scratch/cloned_repositories/torch-summary')
from torchsummary import summary
sys.path.append('/scratch/cloned_repositories/pytorch-msssim')
from pytorch_msssim import msssim

import wandb

import utilities
import models.ae as ae
import models.vae as vae
import data_loader as data_loader
from metrics import metrics  # TODO

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/scratch/image_datasets/3_65x65/ready',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='models/',
                    help="Directory containing params.json")
parser.add_argument('--weights_dir', default='/scratch/image_datasets/3_65x65/ready/weights',
                    help="Directory where weights will be saved")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'


def train_epoch(model, optimizer, loss_fn, dataloader, metrics, params):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utilities.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, train_batch in enumerate(dataloader):
            # move to GPU if available
            if params.cuda:
                train_batch = train_batch.cuda(non_blocking=True)
            # convert to torch Variables
            train_batch = Variable(train_batch)

            # compute model output and loss
            if params.variational:
                output_batch, mu, logvar = model(train_batch)
                loss = - loss_fn(output_batch, train_batch, mu, logvar)
            else:
                output_batch = model(train_batch)
                loss = - loss_fn(output_batch, train_batch)
            # pdb.set_trace()


            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                # output_batch = output_batch.data.cpu().numpy()

                # compute all metrics on this batch
                summary_batch = {metric: metrics[metric](output_batch, train_batch)
                                 for metric in metrics}
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # pdb.set_trace()

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)
    return metrics_mean


def evaluate_epoch(model, loss_fn, dataloader, metrics, params):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []

    # compute metrics over the dataset
    for data_batch in dataloader:

        # move to GPU if available
        if params.cuda:
            data_batch = data_batch.cuda(non_blocking=True)
        # fetch the next evaluation batch
        data_batch = Variable(data_batch)

        # compute model output
        if params.variational:
            output_batch, mu, logvar = model(data_batch)
            loss = - loss_fn(output_batch, data_batch, mu, logvar)
        else:
            output_batch = model(data_batch)
            loss = - loss_fn(output_batch, data_batch)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        # output_batch = output_batch.data.cpu().numpy()
        # data_batch = data_batch.data.cpu().numpy()

        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](output_batch, data_batch)
                         for metric in metrics}
        summary_batch['loss'] = loss.item()
        summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, metrics, params, model_dir,
                       weights_dir, restore_file=None, use_wandb=True):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config and log
        weights_dir: (string) directory containing weights
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(
            args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utilities.load_checkpoint(restore_path, model, optimizer)

    best_val_loss = math.inf  # might need to change (to 0.0) if changing the metric

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train_metrics = train_epoch(model, optimizer, loss_fn, train_dataloader, metrics, params)
        train_loss, train_mse = train_metrics['loss'], train_metrics['mse']  # TODO generalise this

        # Evaluate for one epoch on validation set
        val_metrics = evaluate_epoch(model, loss_fn, val_dataloader, metrics, params)
        val_loss, val_mse = val_metrics['loss'], val_metrics['mse']  # TODO generalise this
        is_best = val_loss <= best_val_loss  # might need to change (to >=) if changing the metric

        # Save weights
        utilities.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                                  is_best=is_best,
                                  checkpoint=weights_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best loss")
            best_val_loss = val_loss

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(
                weights_dir, "metrics_val_best_weights.json")
            utilities.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(
            weights_dir, "metrics_val_last_weights.json")
        utilities.save_dict_to_json(val_metrics, last_json_path)

        if use_wandb:
            wandb.log({"loss": train_loss, "val_loss": val_loss, "mse": train_mse, "val_mse": val_mse})


if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utilities.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    weights_filename_suffix = 'vae' if params.variational else 'ae'
    weights_dir = os.path.join(args.weights_dir, "weights_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")) + "_" + weights_filename_suffix
    Path(weights_dir).mkdir(parents=True, exist_ok=True)

    # Set the logger
    utilities.set_logger(os.path.join(weights_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(
        ['train', 'validation'], args.data_dir, params)
    train_dl = dataloaders['train']
    val_dl = dataloaders['validation']

    logging.info("- done.")

    if params.variational:
        params.beta = params.vae_beta_norm * 32  # 32 = input size / latent size; TODO generalise it
        model = vae.BetaVAE(latent_size=params.latent_size, beta=params.beta).cuda() if params.cuda else vae.BetaVAE(latent_size=params.latent_size, beta=params.beta)
    else:
        model = ae.AE(latent_size=params.latent_size).cuda() if params.cuda else ae.AE(latent_size=params.latent_size)

    # print(model)
    summary(model, (1, 64, 64))
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    use_wandb = False

    if use_wandb:
        wandb.login()
        wandb_run = wandb.init(project="vae-descr", config=params)  # TODO wandb project name should be a parameter
        wandb.watch(model)

    # loss_fn = model.loss
    loss_fn = msssim()  # TODO: figure out how to use relu normalisation

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_dl, val_dl, optimizer, loss_fn, metrics, params, args.model_dir,
                       weights_dir, args.restore_file, use_wandb)

    if use_wandb:
        wandb_run.finish()
