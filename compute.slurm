#!/bin/bash

#SBATCH --job-name=mmg-compute # Job name for tracking
#SBATCH --partition=gpu-batch  # Partition you wish to use (see above for list)
#SBATCH --cpus-per-task=40      # Number of CPU threads used by your job, set this upto 40 as required.
#SBATCH --mem-per-cpu=1500     # RAM in MB needed per core (i.e. 1500MB * 40 threads = 60GB RAM)
#SBATCH --time=2-00:00:00      # Job time limit set to 2 days (48 hours)
#
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80 # Events to send email on, remove if you don't want this
#SBATCH --output=joboutput_%j.out # Standard out from your job
#SBATCH --error=joboutput_%j.err  # Standard error from your job

source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh
srun ./run.sh

