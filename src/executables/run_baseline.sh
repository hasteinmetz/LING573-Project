#!/bin/sh

source ~/anaconda3/etc/profile.d/conda.sh
conda activate /projects/assigned/2122_ling573_elibales/env/573-project

python ../src/baseline.py --train_sentences ../src/data/hahackathon_prepo1_train.csv --dev_sentences ../src/data/hahackathon_prepo1_dev.csv --hidden_layer 100 --learning_rate 0.0001 --batch_size 64 --epochs 15 --random_seeds ../src/data/random_seeds_100.txt --output_file ../outputs/D2/d2_out.txt
