#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --mem=3G
#SBATCH --job-name=avg_sent_len
#SBATCH --output=log/%x_%j.out

# Check the length of the sentences in the training and test data (Nodalida paper section 4.2)

for seed in 11 22 33 44 55 66 77 88
do
    wc_out_comdiv0=$(wc exp/subset-d-1m/splits/ranges0-1000000-1000-auto-40000.10_comweight34/comdiv0.0_seed${seed}_subsample1000every2iters_testsize0.2to0.3_leaveout0.0/iter200000/raw_fi_test_full.txt)
    wc_out_comdiv1=$(wc exp/subset-d-1m/splits/ranges0-1000000-1000-auto-40000.10_comweight34/comdiv1.0_seed${seed}_subsample1000every2iters_testsize0.2to0.3_leaveout0.0/iter200000/raw_fi_test_full.txt)

    avg_words_per_sent_comdiv0=$(echo $wc_out_comdiv0 | awk '{print $2/$1}')
    avg_words_per_sent_comdiv1=$(echo $wc_out_comdiv1 | awk '{print $2/$1}')

    echo "Test, seed $seed: avg_words_per_sent_comdiv0 $avg_words_per_sent_comdiv0, avg_words_per_sent_comdiv1 $avg_words_per_sent_comdiv1"

    wc_out_comdiv0=$(wc exp/subset-d-1m/splits/ranges0-1000000-1000-auto-40000.10_comweight34/comdiv0.0_seed${seed}_subsample1000every2iters_testsize0.2to0.3_leaveout0.0/iter200000/raw_fi_train.txt)
    wc_out_comdiv1=$(wc exp/subset-d-1m/splits/ranges0-1000000-1000-auto-40000.10_comweight34/comdiv1.0_seed${seed}_subsample1000every2iters_testsize0.2to0.3_leaveout0.0/iter200000/raw_fi_train.txt)

    avg_words_per_sent_comdiv0=$(echo $wc_out_comdiv0 | awk '{print $2/$1}')
    avg_words_per_sent_comdiv1=$(echo $wc_out_comdiv1 | awk '{print $2/$1}')

    echo "Train, seed $seed: avg_words_per_sent_comdiv0 $avg_words_per_sent_comdiv0, avg_words_per_sent_comdiv1 $avg_words_per_sent_comdiv1"
done

