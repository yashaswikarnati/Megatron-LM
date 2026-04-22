#!/bin/bash
#SBATCH --job-name=rce-proof
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:05:00
python tests/test_safe_unpickler_rce_proof.py
