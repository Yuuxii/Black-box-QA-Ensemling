#!/bin/bash
#BATCH --nodes=1               
#SBATCH --job-name=mt_viz    ## name you give to your job
#SBATCH --cpus-per-task=4       ## number of cores for the job - max 80
#SBATCH --time=168:00:00
#SBATCH --mem=16G                ## total memory for the job - max 512G
#SBATCH --partition=p_csunivie_gres
#SBATCH --gres=gpu:1           ## number of GPUs for the job - max :8         
#SBATCH --output=outs/mt_viz.out  ## sysout and syserr merged together

export PYTHONPATH=$PYTHONPATH:src
python src/tasks/infosel_mt_vqa.py --multiGPU --output ./snap/viz_models --config ./configs/vizwiz.json --dataname vizwiz --load ./snap/vqa --features vqa 



