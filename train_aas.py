# Load environment variables
from dotenv import load_dotenv
load_dotenv(override=True)

# Import modules
from aas import config as cfg
from aas.env import AASTrainer
from utils.log import print_info
from utils.neptune_utils import init_neptune
import os

# Set trainer
trainer = AASTrainer(env_type='op', save_file_path=os.getenv('AAS_SAVE_FILE_PATH', None))
print_info('Env build ...')
print_info(f'Temporary training env file at {trainer.train_env.tmp_file_path}')
print_info(f'Temporary evaluation env file at {trainer.eval_env.tmp_file_path}')
print_info(f"Agent's network saved at {trainer.save_file_path}")
# Print configuration
print_info('Configuration:')
print_info(cfg)

# Set neptune logs if enabled
neptune_logs = init_neptune(['aas'] + cfg.tags, cfg_path=os.getenv('AAS_CONFIG_FILE_PATH'), mode='sync') if cfg.logging else None

# Start training
print_info('Start training ... ')
trainer.train(neptune_logs=neptune_logs)

# Stop logs if enabled
if cfg.logging:
    neptune_logs.stop()

print_info('Training ended ... ')
