#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --mem=156G
#SBATCH --job-name=significance_tests
#SBATCH --output=log/%x_%j.out

ref_target=$1
pred_target_baseline=$2
pred_target_system=$3

if [ -z "$ref_target" ] || [ -z "$pred_target_baseline" ] || \
    [ -z "$pred_target_system" ]; then
    echo "Usage: $0 <ref_target> <pred_target_baseline> <pred_target_system>"
    exit 1
fi

(set -x; sacrebleu \
    "$ref_target" \
    --width 4 \
    --input "$pred_target_baseline" "$pred_target_system" \
    --metrics bleu chrf --chrf-word-order 2 \
    --paired-bs --paired-bs-n 3000 \
    > "${pred_target_system%.detok}.paired-bs") || exit 1

(set -x; sacrebleu \
    "$ref_target" \
    --width 4 \
    --input "$pred_target_baseline" "$pred_target_system" \
    --metrics bleu chrf --chrf-word-order 2 \
    --paired-ar \
    > "${pred_target_system%.detok}.paired-ar") || exit 1
