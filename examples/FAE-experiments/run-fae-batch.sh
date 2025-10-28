#!/bin/bash
#SBATCH --job-name=fae-batch
#SBATCH --output=lasdi_%A_%a.out
#SBATCH --error=lasdi_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=00:03:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=pdebug

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate conda environment
source ~/.bashrc
conda activate python3.9
cd /usr/workspace/trautner/GPLaSDI/examples/FAE-experiments

export PYTHONUNBUFFERED=1

# Define parameter arrays
max_iters=(2000 5000 10000)
pointwise_lifts=(5 20)
layer_widths=(25 50)

# Calculate total combinations: 3 * 2 * 2 = 12 (indices 0-11)
n_max_iters=${#max_iters[@]}
n_pointwise_lifts=${#pointwise_lifts[@]}
n_layer_widths=${#layer_widths[@]}

# Calculate indices for this array task
mi_idx=$((SLURM_ARRAY_TASK_ID / (n_pointwise_lifts * n_layer_widths)))
pl_idx=$(((SLURM_ARRAY_TASK_ID / n_layer_widths) % n_pointwise_lifts))
lw_idx=$((SLURM_ARRAY_TASK_ID % n_layer_widths))

# Get parameter values
max_iter=${max_iters[$mi_idx]}
pointwise_lift=${pointwise_lifts[$pl_idx]}
layer_width=${layer_widths[$lw_idx]}

# Construct config filename
config_dir="burgers1d-FAE-MI${max_iter}-PLD${pointwise_lift}-LW${layer_width}"
config_file="${config_dir}/config.yaml"

echo "================================================"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Running with parameters:"
echo "  max_iter: $max_iter"
echo "  pointwise_lift: $pointwise_lift"
echo "  layer_width: $layer_width"
echo "  config_file: $config_file"
echo "================================================"



# Run the command
lasdi "$config_file"

echo "Job completed successfully"