#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --mem=22G
#SBATCH --job-name=significance_tests
#SBATCH --output=log/%x_%j.out

exp_name=$1
datadir=$2
src_lang=$3
tgt_lang=$4
vocab_size_src_baseline=$5
vocab_size_src_system=$6
vocab_size_tgt=$7
if [ -z "$exp_name" ] || [ -z "$datadir" ] || [ -z "$src_lang" ] || \
    [ -z "$tgt_lang" ] || [ -z "$vocab_size_src_baseline" ] || \
    [ -z "$vocab_size_src_system" ] || [ -z "$vocab_size_tgt" ]; then
    echo "Usage: $0 <exp_name> <datadir> <src_lang> <tgt_lang> \\"
    echo "   <vocab_size_src_baseline> <vocab_size_tgt_baseline> \\"
    echo "   <vocab_size_tgt_system>"
    exit 1
fi
. ./exp/${exp_name}/config.sh

nmt_dir_baseline="${datadir}/nmt-${src_lang}-${tgt_lang}_vocabs_${vocab_size_src_baseline}_${vocab_size_tgt}"
nmt_dir_system="${datadir}/nmt-${src_lang}-${tgt_lang}_vocabs_${vocab_size_src_system}_${vocab_size_tgt}"

model_cp_baseline=$(cat "${nmt_dir_baseline}/best_model_cp.txt")
model_cp_system=$(cat "${nmt_dir_system}/best_model_cp.txt")

ref_target="${datadir}/raw_${tgt_lang}_test_full.txt"
pred_target_baseline="${nmt_dir_baseline}/test_pred_cp${model_cp_baseline}_full.txt"
pred_target_system="${nmt_dir_system}/test_pred_cp${model_cp_system}_full.txt"

echo "exp_name: ${exp_name}"
echo "datadir: ${datadir}"
echo "src_lang: ${src_lang}"
echo "tgt_lang: ${tgt_lang}"
echo "reference target: $ref_target"
echo "predicted target baseline: $pred_target_baseline"
echo "predicted target system: $pred_target_system"

if [ ! -f "${pred_target_baseline%.txt}.detok" ]; then
    (set -x; python utils/postprocess_translations.py \
        "${ref_target}" \
        "${pred_target_baseline}" \
        "${pred_target_baseline%.txt}.detok" \
        --parse-aligned \
        --desubword-hyp) || exit 1
fi
if [ ! -f "${pred_target_system%.txt}.detok" ]; then
    (set -x; python utils/postprocess_translations.py \
        "${ref_target}" \
        "${pred_target_system}" \
        "${pred_target_system%.txt}.detok" \
        --parse-aligned \
        --desubword-hyp) || exit 1
fi

(set -x; sacrebleu \
    "$ref_target" \
    --width 4 \
    --input "${pred_target_baseline%.txt}.detok" \
        "${pred_target_system%.txt}.detok" \
    --metrics bleu chrf --chrf-word-order 2 \
    --paired-bs --paired-bs-n 3000 \
    > "${pred_target_system%.txt}.paired-bs") || exit 1

(set -x; sacrebleu \
    "$ref_target" \
    --width 4 \
    --input "${pred_target_baseline%.txt}.detok" \
        "${pred_target_system%.txt}.detok" \
    --metrics bleu chrf --chrf-word-order 2 \
    --paired-ar \
    > "${pred_target_system%.txt}.paired-ar") || exit 1

if [ -z "$SLURM_JOB_ID" ]; then
    exit 0
fi
if [ -f "log/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" ]; then
    sleep 20 # wait for the file to be written
    cp "log/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" \
        "${nmt_dir_system}/" || true
fi
