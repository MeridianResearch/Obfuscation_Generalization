#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

# Evaluation dispatcher with Hydra config support
#
# Usage:
#   sbatch --job-name=eval-sycophancy eval_dispatch.sh \
#       experiment=full_xml_tags/eval_sycophancy \
#       training_group=leave_out_sycophancy_full_xml_tags_seed_42 \
#       training_run_name=monitor_informed_pen \
#       artifact_step=100
#
# Environment variables (optional):
#   OBF_GEN_CONDA_ENV  - Conda environment (default: obf_gen)
#   OBF_GEN_WORKDIR    - Working directory (default: ~/repos/Obfuscation_Generalization)

# All arguments are Hydra overrides
HYDRA_OVERRIDES="$@"

if [ -z "$HYDRA_OVERRIDES" ]; then
    echo "Error: Hydra overrides are required"
    echo "Usage: sbatch eval_dispatch.sh <hydra_overrides>"
    echo ""
    echo "Required overrides:"
    echo "  experiment=<experiment_config>"
    echo "  training_group=<wandb_group>"
    echo "  training_run_name=<wandb_run_name>"
    echo "  artifact_step=<checkpoint_step>"
    echo ""
    echo "Example:"
    echo "  sbatch eval_dispatch.sh experiment=full_xml_tags/eval_sycophancy \\"
    echo "      training_group=leave_out_sycophancy_full_xml_tags_seed_42 \\"
    echo "      training_run_name=monitor_informed_pen \\"
    echo "      artifact_step=100"
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

python -m src.eval $HYDRA_OVERRIDES

echo ""
echo "End time: $(date)"
