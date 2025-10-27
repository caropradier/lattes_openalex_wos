#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --account=def-vlarivie
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=embeddings
#SBATCH --mem=120G
#SBATCH --gpus-per-node=2
#SBATCH --mail-user=carolina.pradier@umontreal.ca
#SBATCH --mail-type=ALL
#SBATCH  --output="../../job_outputs/job-%u-%x-%j.out"

# ---------------------------------------------------------------------
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
# ---------------------------------------------------------------------
echo ""
echo "Job ID:  $SLURM_JOB_ID"
echo ""
# ---------------------------------------------------------------------
module load python
module load scipy-stack
module load StdEnv/2023
module load gcc arrow
module load java/11.0.22
source ~/ENV/bin/activate

partition=${1:-1}  # Use ${1:-default_value} for default assignment

python 1_embedding.py $partition
# ---------------------------------------------------------------------
echo "Job finished with exit code $? at: `date`"
# ---------------------------------------------------------------------
