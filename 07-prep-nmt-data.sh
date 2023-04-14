#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=3G
#SBATCH --job-name=prep_nmt_data
#SBATCH --output=log/%x_%j.out

exp_name=$1
datadir=$2
train_set=$3
test_set=$4
output_dir=$5
if [ -z "$exp_name" ] || [ -z "$datadir" ] || [ -z "$train_set" ] \
    || [ -z "$test_set" ] || [ -z "$output_dir" ]; then
    echo "Usage: $0 <exp_name> <datadir> <train_set> <test_set> <output_dir>"
    exit 1
fi
. ./exp/${exp_name}/config.sh

if [ ! -d "$output_dir" ]; then
    (set -x; mkdir -p "$output_dir")
else
    echo "$output_dir exists, exiting"
    exit 1
fi

train_set_name=$(basename $train_set)
train_augmented="${output_dir}/${train_set_name%.txt}.augmented"
if [ ! -f "$train_augmented" ]; then
    echo "Augmenting train set with ids in"
    echo "${datadir}/unused_sent_ids.txt"
    if [[ -s "$1" && -z "$(tail -c 1 "$1")" ]]
    then
        true
    else
        echo "" >> "${datadir}/unused_sent_ids.txt"
    fi
    cat "${datadir}/unused_sent_ids.txt" "$train_set" > "$train_augmented" || exit 1
    echo "train set size before augmentation: $(wc -l $train_set)"
    echo "train set size after augmentation: $(wc -l $train_augmented)"
else
    echo "$train_augmented exists, skipping augmentation."
fi

args=("$train_augmented" \
    "$test_set" \
    "$filtered_fi_data" \
    "$filtered_en_data" \
    "$output_dir")
for file in "${args[@]}"; do
    if [ ! -f "$file" ] && [ ! -d "$file" ]; then
        echo "File $file does not exist"
        exit 1
    fi
done
if [ -f "$line2original_file" ]; then
    args+=("--line2original" "$line2original_file")
fi

(set -x; python prep_onmt_data.py "${args[@]}") || exit 1
