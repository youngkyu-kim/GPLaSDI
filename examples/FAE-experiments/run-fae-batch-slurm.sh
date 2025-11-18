#!/bin/bash
#SBATCH --job-name=fae-batch-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=logs/lasdi_%a.out
#SBATCH --error=logs/lasdi_%a.err
#SBATCH --array=0

echo "Script starting..."

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate conda environment
source ~/.bashrc

echo "Activated conda environment"

cd /resnick/groups/astuart/trautner/LLNL/GPLaSDI/examples/FAE-experiments


export PYTHONUNBUFFERED=1

# Define parameter arrays
max_iters=(2000 5000 10000)
pointwise_lifts=(5)
layer_widths=(25 50)

# Calculate total combinations: 3 * 1 * 2 = 6 (indices 0-5)
n_max_iters=${#max_iters[@]}
n_pointwise_lifts=${#pointwise_lifts[@]}
n_layer_widths=${#layer_widths[@]}

# Use SLURM_ARRAY_TASK_ID
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

# Calculate indices for this array task
mi_idx=$((TASK_ID / (n_pointwise_lifts * n_layer_widths)))
pl_idx=$(((TASK_ID / n_layer_widths) % n_pointwise_lifts))
lw_idx=$((TASK_ID % n_layer_widths))

# Get parameter values
max_iter=${max_iters[$mi_idx]}
pointwise_lift=${pointwise_lifts[$pl_idx]}
layer_width=${layer_widths[$lw_idx]}

# Construct config filename
config_dir="burgers1d-FAE-MI${max_iter}-PLD${pointwise_lift}-LW${layer_width}"
config_file="${config_dir}/config.yaml"

echo "================================================"
echo "SLURM_ARRAY_TASK_ID: $TASK_ID"
echo "Running with parameters:"
echo "  max_iter: $max_iter"
echo "  pointwise_lift: $pointwise_lift"
echo "  layer_width: $layer_width"
echo "  config_file: $config_file"
echo "================================================"


# Run the command
lasdi "$config_file"

echo "Job completed"