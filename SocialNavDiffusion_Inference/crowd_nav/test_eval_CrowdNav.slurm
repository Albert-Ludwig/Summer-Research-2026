#!/bin/bash
#SBATCH --job-name=training         # Job name
#SBATCH --output=logs/robot_CrowdNav_%j.out    # STDOUT log
#SBATCH --error=logs/robot_CrowdNav_%j.err     # STDERR log
#SBATCH --time=3:00:00                    # Max runtime (HH:MM:SS)
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --ntasks-per-node=1                # Number of tasks per node
#SBATCH --cpus-per-task=16                 # CPU cores per task
#SBATCH --mem=64G                          # Memory
#SBATCH --gres=gpu:1                        # Number of GPUs
#SBATCH --partition=gpubase_l40s_b1                     # GPU partition
#SBATCH --mail-type=END,FAIL               # Email notification
#SBATCH --mail-user=cschaibl@uwaterloo.com  # Your email

# --------------------------
# Load modules / activate env
# --------------------------
module load python/3.10
module load cuda/12.6
module load StdEnv/2020
module load cudacore/.11.2.2
module load cudnn/8.2.0

# Activate your virtual environment
source ~/projects/aip-sl2smith/cschaibl/venvTHROWAWAY/bin/activate

export ACADOS_SOURCE_DIR=/home/cschaibl/projects/aip-sl2smith/cschaibl/diffusion_planning/acados
export LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$ACADOS_SOURCE_DIR/interfaces/acados_template:$PYTHONPATH

# Make logs/checkpoints directories
mkdir -p logs
mkdir -p checkpoints

cd ~/projects/aip-sl2smith/cschaibl/diffusion_planning/CrowdNav_DiffusionEnv/crowd_nav
# --------------------------
# Run the training script
# --------------------------

TEST_NAME="GC CLEAN"

python test_evaluate.py \
    --policy diffusion_conditional_unet1dcfg \
    --model_dir data/output \
    --phase test \
    --visualize \
    --test_case 81 \
    --gpu \
    --video_file "${TEST_NAME}.mp4" \
    --results_suffix "baselines" \
    # --map_eval \
    # --npz_hard \
    # --npz_dir /home/cschaibl/projects/aip-sl2smith/cschaibl/diffusion_planning/diffusion_STAGE_B/MatterportMAPS

# --------------------------
# Copy the .err file to results folder
# --------------------------
# ERR_FILE="logs/robot_CrowdNav_${SLURM_JOB_ID}.err"
# if [ -f "$ERR_FILE" ]; then
#     cp "$ERR_FILE" results/${TEST_NAME}_eval.txt
#     echo "Copied $ERR_FILE to results/robot_CrowdNav_${TEST_NAME}_eval.txt"
# else
#     echo "Error file not found: $ERR_FILE"
# fi