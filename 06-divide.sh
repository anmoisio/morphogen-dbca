#!/bin/bash
#SBATCH --time=34:00:00
#SBATCH --partition dgx-spa,gpu,dgx-common
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --constraint=volta
#SBATCH --job-name=divide
#SBATCH --output=log/%x_%j.out

exp_name=$1
data_dir=$2
comdiv=$3
random_seed=$4
if [ -z "$exp_name" ] || [ -z "$data_dir" ] || [ -z "$comdiv" ]; then
    echo "Usage: $0 <exp_name> <data_dir> <comdiv> [<random_seed>]"
    exit 1
fi
. ./exp/${exp_name}/config.sh

if [ -z "$random_seed" ]; then
    random_seed=11
fi

(set -x; python divide.py \
    --data-dir $data_dir \
    --min-test-percent $min_test_percent \
    --max-test-percent $max_test_percent \
    --subsample-size $subsample_size \
    --subsample-iter $subsample_iter \
    --max-iters $max_iters \
    --save-cp $save_cp \
    --print-every $print_every \
    --random-seed $random_seed \
    --compound-divergences $comdiv) || exit 1

if [ -f log/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out ]; then
    sleep 10
    cp log/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out $data_dir/ || true
fi
