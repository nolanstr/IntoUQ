#!/bin/bash -l

#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --partition=defq
#SBATCH --account=sghosh20-durip

source ~/.bashrc

python ../main_script.py
