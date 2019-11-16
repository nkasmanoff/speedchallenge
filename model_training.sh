#!/bin/bash
#
##SBATCH --nodes=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=6:00:00
#SBATCH --mem=24GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=self_driving_car
#SBATCH --mail-type=END
##SBATCH --mail-user=nsk367@nyu.edu
#SBATCH --output=slurm_%j.out


python train.py
