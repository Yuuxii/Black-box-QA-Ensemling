#!/bin/bash
#BATCH --nodes=1               ## always 1 (we have just 1 server)
#SBATCH --job-name=infosel_tt_sq_no_chatgpt    ## name you give to your job
#SBATCH --cpus-per-task=4       ## number of cores for the job - max 80
#SBATCH --mem=16G                ## total memory for the job - max 512G
#SBATCH --gres=gpu:1           ## number of GPUs for the job - max :8         
#SBATCH --output=outs/infosel_tt_sq_no_chatgpt.out  ## sysout and syserr merged together
 
conda activate llm     

python main/infosel_tt_qa.py --dataname sq --use_amount 1000 --output_dir tt_outs/sq/1k_no_chatgpt  --lr 5e-5 --exclude_worst_model

# python main/infosel_tt_qa.py --dataname sq --use_amount 1000 --output_dir tt_outs/sq/1k  --lr 5e-5 




