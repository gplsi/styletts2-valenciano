#!/bin/bash
#SBATCH --job-name=style_tts_multinode
#SBATCH --output=./Logs/%j.out
#SBATCH --error=./Logs/%j.err
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=2               # Increased to 2 nodes
#SBATCH --ntasks-per-node=1     # One task per node (Accelerate handles GPUs internally)
#SBATCH --gres=gpu:4            # 4 GPUs per node
#SBATCH --cpus-per-task=16      # Increased for better data loading
#SBATCH --mem=64G               # Increased for multi-GPU overhead
#SBATCH --time=72:00:00

# 1. Environment Setup
source /leonardo_work/EUHPC_D22_034/miniconda3/etc/profile.d/conda.sh
conda activate Styletts

# 2. Networking for Multi-Node Accelerate
# Get the IP of the first node as the Master
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * 4)) # Total GPUs = Nodes * 4

# NCCL Optimizations for high-speed clusters
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export ACCELERATE_DEBUG_MODE=1

# 3. Informational output
nvidia-smi
echo "Master Node: $MASTER_ADDR"
echo "Total GPUs (World Size): $WORLD_SIZE"

# 4. Multi-Node Launch
# Use srun to trigger the command on each node
srun accelerate launch \
    --multi_gpu \
    --num_machines $SLURM_NNODES \
    --num_processes $WORLD_SIZE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $SLURM_PROCID \
    --rdzv_backend c10d \
    train_first.py --config_path Configs/config_valenciano.yml
