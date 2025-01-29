# Load environment variables
from dotenv import load_dotenv
load_dotenv(override=True)

# Import modules
from aas import config as cfg
from aas.env import AASTrainer
from aas.agent import AlphaAutoScheduler
from utils.log import print_info
from utils.neptune_utils import init_neptune
from tqdm import tqdm
import os

# Set trainer
trainer = AASTrainer()
print_info('Env build ...')
print_info(f'temporary file path = {trainer.tmp_file_path}')
# Print configuration
print_info('Configuration:')
print_info(cfg)

# Define agent
agent = AlphaAutoScheduler(trainer.tmp_file_path)

# Set neptune logs if enabled
neptune_logs = init_neptune(['aas'] + cfg.tags, cfg_path=os.getenv('AAS_CONFIG_FILE_PATH')) if cfg.logging else None

# Start training
print_info('Start training ... ')

tqdm_range = tqdm(range(cfg.nb_iterations), desc='Main loop')

state = trainer.reset()
for step in tqdm_range:
    state, terminated, speedup = trainer.step(agent, state)
    if cfg.logging:
        print(f'step = {step}, terminated = {terminated}, speedup = {speedup}')
        if terminated:
            neptune_logs['train/final_speedup'].append(speedup)
        print('Selection loss:', agent.network_manager.stats.selection_loss)
        neptune_logs['train/selection_loss'].extend(agent.network_manager.stats.selection_loss)
        print('Parallel params loss:', agent.network_manager.stats.parallel_params_loss)
        for i in range(cfg.max_num_loops):
            neptune_logs[f'train/parallel_params_loss_{i}'].extend(agent.network_manager.stats.parallel_params_loss[i])
        print('Value loss:', agent.network_manager.stats.value_loss)
        neptune_logs['train/value_loss'].extend(agent.network_manager.stats.value_loss)

# Stop logs if enabled
if cfg.logging:
    neptune_logs.stop()

print_info('Training ended ... ')
