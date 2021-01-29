import wandb
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
            'learning_rate': {
                'values': [0.1, 0.01, 0.001, 0.0001, 0.00001]
            },
            'latent_size': {
                'values': [32, 64, 128]
            },
            'batch_size': {
                'values': [32, 64, 128]
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="local-img-descr-ae")  # TODO!!!
    wandb.agent(sweep_id, function=training_sweep)
