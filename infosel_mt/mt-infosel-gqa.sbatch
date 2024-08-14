#!/bin/bash
#BATCH --nodes=1               
#SBATCH --job-name=mt_gqa    ## name you give to your job
#SBATCH --cpus-per-task=4       ## number of cores for the job - max 80
#SBATCH --time=168:00:00
#SBATCH --mem=16G                ## total memory for the job - max 512G
#SBATCH --partition=p_csunivie_gres
#SBATCH --gres=gpu:1           ## number of GPUs for the job - max :8         
#SBATCH --output=outs/mt_gqa_test.out  ## sysout and syserr merged together


# python src/tasks/infosel_mt_vqa.py --multiGPU --output ./snap/gqa_models --config ./configs/gqa.json --dataname gqa --load ./snap/vqa --features vqa 
export PYTHONPATH=$PYTHONPATH:src
python src/tasks/infosel_mt_vqa.py --multiGPU --output ./snap/gqa_models --config ./configs/gqa.json --dataname gqa --load ./snap/gqa_models/BEST --features vqa --test test