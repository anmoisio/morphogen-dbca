#!/bin/bash
#SBATCH --time=3-00
#SBATCH --mem=88G
#SBATCH --job-name=opusfilter
#SBATCH --output=log/%x_%j.out

yaml_file=$1
all_args=("$@")
args="${all_args[@]:1}"
if [ ! -f "$yaml_file" ]; then
    echo "Usage: $0 <yaml_file> [<args to opusfilter>]"
    echo "e.g. $0 exp/full-b/opusfilter_classifier.yaml --n-jobs 8 --single 13"
    exit 1
fi

(set -x; opusfilter $args $yaml_file)
