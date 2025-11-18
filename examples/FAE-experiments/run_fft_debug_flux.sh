#!/bin/bash
echo "Script starting..."

# to run the script: 
# flux submit -N 1 -n 1 -c 1 --gpus-per-task=1 --setattr=system.duration=300 --job-name=fae-batch-gpu --output=fft-debug_{cc}.out --error=fft-debug_{cc}.err --env=PYTHONUNBUFFERED=1 --cc=0 bash run_fft_debug_flux.sh


# Activate conda environment
source ~/.bashrc
conda activate /usr/WS1/trautner/envs/python3.9

echo "Changed conda environment"

cd /usr/workspace/trautner/GPLaSDI/examples/FAE-experiments

export PYTHONUNBUFFERED=1

# Run the command
python fft_debug.py

echo "Job completed successfully"