#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --mem=4G
#SBATCH --job-name=prep_divide
#SBATCH --output=log/%x_%j.out

# options
exp_name=
stage=0
stop_after_stage=10
ranges=
com_weight_threshold=
overwrite=false
weight_compounds=false
profile=false
. ./utils/parse_options.sh

. ./exp/${exp_name}/config.sh

args="$parsed_data"

splitdirname="ranges${ranges}_comweight${com_weight_threshold#0.}"
output_dir="exp/${exp_name}/splits/${splitdirname}"
args="$args $output_dir"

args="$args --stage $stage --stop_after_stage $stop_after_stage"

for arg in com_weight_threshold ranges; do
    if [ ! -z "${!arg}" ]; then
        args="$args --$arg ${!arg}"
    fi
done

for arg in overwrite weight_compounds profile; do
    if [ "${!arg}" = true ]; then
        args="$args --$arg"
    fi
done

(set -x; python prepare_divide_data.py $args) || exit 1

if [ -f log/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out ]; then
    sleep 50
    cp log/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out $output_dir/ || true
fi
