#!/bin/bash
#SBATCH --time=0:30:00
#SBATCH --account=def-vlarivie
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=model_merge
#SBATCH --mem=186G
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


python 3_merging.py  ../../job_outputs/topic_model_update 

# ---------------------------------------------------------------------
echo "Job finished with exit code $? at: `date`"
# ---------------------------------------------------------------------
