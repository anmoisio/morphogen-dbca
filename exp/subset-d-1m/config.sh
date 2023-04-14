#!/bin/bash

root_dir="/scratch/elec/morphogen"
exp_dir="${root_dir}/dbca/exp/${exp_name}"
exp_data_dir="${exp_dir}/data"
mkdir -p "$exp_data_dir"

line2original_file="${exp_data_dir}/sent_ids.txt"
filtered_fi_data="${exp_data_dir}/sents.fi.txt"
filtered_en_data="${exp_data_dir}/sents.en.txt"
parsed_data="${exp_data_dir}/parsed.txt.gz"

# train/test split
min_test_percent="0.2"
max_test_percent="0.3"
subsample_size="1000"
subsample_iter="2"
max_iters="9999999999"
save_cp="10000"
print_every="1000"

# sentencepiece
character_coverage="1.0"
model_type="bpe"
spm_trainset_size="1000000"

# NMT
nmt_n_sample="-1"
save_checkpoint_steps="3000"
keep_checkpoint="15"
seed="3435"
train_steps="33000"
valid_steps="3000"
warmup_steps="2000"
report_every="100"
