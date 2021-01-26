import wandb
from training_sweep import training_sweep

if __name__ == '__main__':
    sweep_config = {
        'method': 'grid',
        'project': 'temp',
        'metric': {
            'goal': 'minimize',
            'name': 'loss'
        },
        'parameters': {
            'learning_rate': {
                'values': [0.002, 0.0002]
            },
            'temp_param': {
                'values': [1, 2]
            }
        }
    }
    # wandb.login()

    sweep_id = wandb.sweep(sweep_config, project="temp")
    # sweep_id = '1k2g57ou'
    wandb.agent(sweep_id, function=training_sweep)  # project='temp'