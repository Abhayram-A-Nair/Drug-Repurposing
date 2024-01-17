#!/bin/bash
#SBATCH -A research
#SBATCH -n 39
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --output=./run_log.txt
#SBATCH --mail-type=END

python3 calc.py
