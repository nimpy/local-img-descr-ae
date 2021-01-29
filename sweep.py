import wandb
from training_sweep import training_sweep

if __name__ == '__main__':
    sweep_config = {
        'method': 'grid',
        'project': 'temp',
        'metric': {
            'goal': 'minimize',
            'name': 'hpatches_overall'
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

    sweep_id = wandb.sweep(sweep_config, project="temp")
    wandb.agent(sweep_id, function=training_sweep)
