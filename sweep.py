import wandb
import os
from pathlib import Path

import utilities
from training_sweep import training_sweep

if __name__ == '__main__':
    sweep_config = {
        'method': 'grid',
        'project': 'local-img-descr-ae',  # TODO!!!
        'metric': {
            'goal': 'minimize',
            'name': 'hpatches_overall'
        },
        'parameters': {
            'data_augm_level': {
                'values': [1, 2, 3]
            },
            'activation_fn': {
                'values': ['elu', 'relu']
            },
            'loss_fn': {
                'values': ['bce', 'msssim']
            },
            'learning_rate': {
                'values': [0.01, 0.0001]
            }#,
            # 'latent_size': {
            #     'values': [32, 64, 128]
            # },
            # 'batch_size': {
            #     'values': [32, 64, 128]
            # }
        }
    }

    sweep_version = 'sweep_2nd_ae_latent32_batch32'  # TODO change in both files!!! TODO make it a param passed to a sweep agent
    sweep_dir = os.path.join('/scratch/image_datasets/3_65x65/ready/weights', sweep_version)
    Path(sweep_dir).mkdir(parents=True, exist_ok=True)

    # Set the logger
    utilities.set_logger(os.path.join(sweep_dir, 'train.log'))

    sweep_id = wandb.sweep(sweep_config, project="local-img-descr-ae")  # TODO!!!
    wandb.agent(sweep_id, function=training_sweep)
