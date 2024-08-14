#!/bin/bash
#BATCH --nodes=1               ## always 1 (we have just 1 server)
#SBATCH --job-name=ft_tt_nq    ## name you give to your job
#SBATCH --cpus-per-task=4       ## number of cores for the job - max 80
#SBATCH --mem=16G                ## total memory for the job - max 512G
#SBATCH --gres=gpu:1           ## number of GPUs for the job - max :8         
#SBATCH --output=outs/ft_tt_nq.out  ## sysout and syserr merged together
 
conda activate vbert     

python main/ft_nq.py --use_amount 1000 --dataname nq --save_predictions




