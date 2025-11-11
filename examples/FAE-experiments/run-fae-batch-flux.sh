#!/bin/bash

# Flux Job Specifications (as comments for reference)
# Submit this script with:
# flux submit -N 1 -n 1 -c 4 -g 1 --setattr=system.duration=24h \
#   --job-name=fae-batch-gpu --output=logs/lasdi_{cc}.out \
#   --error=logs/lasdi_{cc}.err --env=PYTHONUNBUFFERED=1 \
#   --array=0-5 bash this_script.sh

# OR use flux alloc/run if on pdebug-equivalent
# flux alloc -N 1 -n 1 -c 4 -g 1 -t 5m

# Auto-submit to Flux if not already running as part of job array
# N nodes, n tasks, c cpus, g gpus
 # Unbuffered output
if [ -z "$FLUX_TASK_RANK" ]; then
    echo "Submitting job array to Flux..."
    flux submit -N 1 -n 1 -c 4 -g 1 \
        --setattr=system.duration=24h \
        --job-name=fae-batch-gpu \
        --output=logs/lasdi_{cc}.out \
        --error=logs/lasdi_{cc}.err \
        --env=PYTHONUNBUFFERED=1 \
        --array=0 \
        bash "$0"
    exit 0
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate conda environment
source ~/.bashrc
conda activate python3.9

cd /usr/workspace/trautner/GPLaSDI

# Compile lasdi
pip install .

cd /usr/workspace/trautner/GPLaSDI/examples/FAE-experiments

export PYTHONUNBUFFERED=1

# Define parameter arrays
max_iters=(2000 5000 10000)
pointwise_lifts=(5)
layer_widths=(25 50)

# Calculate total combinations: 3 * 1 * 2 = 6 (indices 0-5)
n_max_iters=${#max_iters[@]}
n_pointwise_lifts=${#pointwise_lifts[@]}
n_layer_widths=${#layer_widths[@]}

# Use FLUX_TASK_RANK instead of SLURM_ARRAY_TASK_ID
TASK_ID=${FLUX_TASK_RANK:-0}

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
echo "FLUX_TASK_RANK: $TASK_ID"
echo "Running with parameters:"
echo "  max_iter: $max_iter"
echo "  pointwise_lift: $pointwise_lift"
echo "  layer_width: $layer_width"
echo "  config_file: $config_file"
echo "================================================"

# Run the command
lasdi "$config_file"

echo "Job completed successfully"