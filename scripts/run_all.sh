#!/bin/bash
# scripts/run_all.sh

echo "======================================================================"
echo "Running All GFAML Research Experiments..."
echo "======================================================================"

python -m experiments.exp_01_baseline
python -m experiments.exp_02_gfaml
python -m experiments.exp_03_ablation
python -m experiments.exp_04_real_data

echo "======================================================================"
echo "All Experiments Completed Successfully!"
echo "======================================================================"
