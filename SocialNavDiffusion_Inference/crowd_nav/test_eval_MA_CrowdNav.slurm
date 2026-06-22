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

# Make logs/checkpoints directories
mkdir -p logs
mkdir -p checkpoints

cd ~/projects/aip-sl2smith/cschaibl/diffusion_planning/CrowdNav_DiffusionEnv/crowd_nav
# --------------------------
# Run the training script
# --------------------------

TEST_NAME="multiagent K samples"

python test_evaluate_MA.py \
    --policy single_agent_proxy \
    --model_dir data/output \
    --phase test \
    --multi_agent \
    --num_agents 2 \
    --visualize \
    --test_case 81 \
    --gpu \
    --video_file "${TEST_NAME}.mp4" \
    --results_suffix "50ep_multiagent" \
    # --npz_hard \

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