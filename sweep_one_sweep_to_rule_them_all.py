import argparse
import logging
import os
from pathlib import Path
import pandas as pd
import wandb


parser = argparse.ArgumentParser()
parser.add_argument('--weights_dir', default='/scratch/image_datasets/3_65x65/ready/weights',
                    help="Directory where weights will be saved")


def load_sweep_csv(filepath):

    # I made this csv file by downloading csv files from different wandb sweeps and uniting them all into one csv file
    df = pd.read_csv(filepath)

    # replace . with _ in the column names so that pandas doesn't rename the columns
    new_columns = df.columns.values
    for i, column in enumerate(new_columns):
        new_column = column.replace('.', '_')
        new_columns[i] = new_column
    df.columns = new_columns
    return df


sweep_df = load_sweep_csv('/home/niaki/Downloads/1Sweep2RuleEmALL.csv')


def find_row_with_inputs(activation_fn, data_augm_level, loss_fn, vae_beta_norm):
    for row in sweep_df.itertuples():
        if row.activation_fn == activation_fn and row.data_augm_level == data_augm_level and \
                 row.loss_fn == loss_fn and row.vae_beta_norm == vae_beta_norm:
            print(row.Name)
            return row


def sweep_one_sweep_to_rule_them_all():

    args = parser.parse_args()

    use_wandb = True  # TODO!

    if use_wandb:
        wandb_run = wandb.init()  # TODO wandb project name should be a parameter

    logging.info("\n\n****************** STARTING A NEW RUN ******************")
    logging.info('Data augmentation level: ' + str(wandb.config.data_augm_level))
    logging.info('Activation function    : ' + str(wandb.config.activation_fn))
    logging.info('Loss function          : ' + str(wandb.config.loss_fn))
    logging.info('Beta value (normalised): ' + str(wandb.config.vae_beta_norm))
    # logging.info('VAE? (if not then AE)  : ' + str(wandb.config.vae))
    # logging.info('Learning rate          : ' + str(wandb.config.learning_rate))
    logging.info("")

    latent_size = 32
    batch_size = 32
    logging.info('Other params (that are not being swept)')
    logging.info('    Latent size:' + str(latent_size))
    logging.info('    Batch size :' + str(batch_size))
    logging.info("")

    wandb.config.variational = wandb.config.vae_beta_norm > 0.0000001
    wandb.config.latent_size = latent_size
    wandb.config.batch_size = batch_size
    wandb.config.num_workers = 4
    wandb.config.vae_or_ae = "vae" if wandb.config.vae_beta_norm > 0.0000001 else "ae"
    # wandb.config.learning_rate == 0.0001
    # wandb.config.num_epochs = 200  #

    sweep_version = 'sweep__one_sweep_to_rule_them_all_v1'  # TODO change in both files!!! TODO make it a param passed to a sweep agent
    # model_version = "weights_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_AEorVAE"
    # weights_dir = os.path.join(args.weights_dir, sweep_version, model_version)

    Path(os.path.join(args.weights_dir, sweep_version)).mkdir(parents=True, exist_ok=True)
    # Path(weights_dir).mkdir(parents=True, exist_ok=True)


    if use_wandb:
        # wandb.watch(model)
        pass


    row_df = find_row_with_inputs(wandb.config.activation_fn, wandb.config.data_augm_level, wandb.config.loss_fn, wandb.config.vae_beta_norm)

    if use_wandb:
        wandb.log({"num_epochs": row_df.num_epochs, "variational": row_df.variational, "hpatches_overall": row_df.hpatches_overall,
                   "matching_overall": row_df.matching_overall, "retrieval_overall": row_df.retrieval_overall, "verification_overall": row_df.verification_overall,
                   "mse": row_df.mse, "psnr": row_df.psnr, "ssim": row_df.ssim, "loss": row_df.loss})  #, "": row_df., "": row_df.

    if use_wandb:
        wandb_run.finish()
