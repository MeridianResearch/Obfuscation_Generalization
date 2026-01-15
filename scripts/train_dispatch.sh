#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G

# Training dispatcher with Hydra config support
#
# Usage:
#   # Basic training with experiment
#   sbatch --job-name=train-full_xml train_dispatch.sh experiment=full_xml_tags/train
#
#   # Training with overseer penalty
#   sbatch --job-name=train-pen train_dispatch.sh experiment=full_xml_tags/train +reward/overseer=standard
#
#   # Training with custom penalty weight
#   sbatch --job-name=train-pen-0.2 train_dispatch.sh experiment=full_xml_tags/train +reward/overseer=standard \
#       reward.funcs.api_overseer_penalty_func.penalty_weight=-0.2
#
# Environment variables (optional):
#   OBF_GEN_CONDA_ENV  - Conda environment (default: obf_gen)
#   OBF_GEN_WORKDIR    - Working directory (default: ~/repos/Obfuscation_Generalization)

# All remaining arguments are Hydra overrides
HYDRA_OVERRIDES="$@"

if [ -z "$HYDRA_OVERRIDES" ]; then
    echo "Error: At least one Hydra override is required (e.g., experiment=full_xml_tags/train)"
    echo "Usage: sbatch train_dispatch.sh <hydra_overrides>"
    echo ""
    echo "Examples:"
    echo "  sbatch train_dispatch.sh experiment=full_xml_tags/train"
    echo "  sbatch train_dispatch.sh experiment=full_xml_tags/train +reward/overseer=standard"
    exit 1
fi

# Defaults
CONDA_ENV="${OBF_GEN_CONDA_ENV:-obf_gen}"
WORKDIR="${OBF_GEN_WORKDIR:-$HOME/repos/Obfuscation_Generalization}"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Hydra Overrides: $HYDRA_OVERRIDES"
echo "Conda Env: $CONDA_ENV"
echo "Workdir: $WORKDIR"
echo "Start time: $(date)"
echo ""

source ~/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
cd "$WORKDIR"

NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")

accelerate launch --multi_gpu --num_processes $NUM_GPUS \
    -m src.train $HYDRA_OVERRIDES

echo ""
echo "End time: $(date)"
