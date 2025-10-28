import numpy as np
import yaml
from lasdi.workflow import *

results_file = 'lasdi_10_21_2025_13_56.npy'
config_path = 'FAE-experiments/burgers1d-flasdi.yml'
data = np.load(results_file, allow_pickle=True).item()
# for key, value in data.items():
#     print(key)

# load config
with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        cfg_parser = InputParser(config, name='main')


trainer, param_space, physics, latent_space, latent_dynamics = initialize_trainer(config, None)

latent_space.load_state_dict(data['latent_space']['autoencoder_param'])

print("Success!")