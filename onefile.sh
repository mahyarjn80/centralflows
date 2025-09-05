#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40GB
#SBATCH --time=40:00:00
#SBATCH --constraint="a100|h100"
#SBATCH -o /mnt/home/jcohen/slurm/output-%j.out

/mnt/home/jcohen/central_flows/env/bin/python onefile.py "$@"
