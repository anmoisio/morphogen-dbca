#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --job-name=select_corpora
#SBATCH --output=log/%x_%j.out

exp_name=$1
if [ -z "$exp_name" ]; then
    echo "Usage: $0 <exp_name>"
    exit 1
fi
. ./exp/${exp_name}/config.sh

(set -x; python utils/select_corpora.py \
    --opus_id_file "$opus_id_file" \
    --opus_src_file "$opus_train_src_file" \
    --opus_tgt_file "$opus_train_tgt_file" \
    --output_path_src "$selected_en_data" \
    --output_path_tgt "$selected_fi_data" \
    --line2original_file "$line2original_file" \
    --excluded_corpora "$excluded_corpora" \
    --dataset_size "$dataset_size")
