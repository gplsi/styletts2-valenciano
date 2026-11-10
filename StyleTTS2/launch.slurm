#!/bin/bash
#SBATCH --job-name=style_tts
#SBATCH --output=./Logs/%j.out
#SBATCH --error=./Logs/%j.err
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4

# Activar entorno
source /leonardo_work/EUHPC_D22_034/miniconda3/etc/profile.d/conda.sh
conda activate Styletts
# Accelerate settings
export ACCELERATE_DEBUG_MODE=1
export NCCL_P2P_DISABLE=1 # avoid hangup
nvidia-smi
echo "GPUs asignadas por Slurm: $CUDA_VISIBLE_DEVICES"

python -c "import soundfile as sf; print('NODE LIB CHECK:', sf.__libsndfile_version__)"

# Launch training with unbuffered output
#accelerate launch --multi_gpu --num_processes=4 train_first.py --config_path Configs/config_valenciano.yml
#accelerate launch train_first.py --config_path Configs/config_valenciano.yml
accelerate launch --mixed_precision="bf16" train_first.py --config_path Configs/config_valenciano.yml
