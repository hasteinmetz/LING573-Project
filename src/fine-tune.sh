#!/bin/sh

source ~/anaconda3/etc/profile.d/conda.sh
conda activate 573-project

python src/fine-tune.py --train_sentences src/data/hahackathon_prepo1_train.csv --dev_sentences src/data/hahackathon_prepo1_dev.csv --learning_rate 5e-5 --batch_size 64 --epochs 1 --output_file outputs/test/fine-tune_results.txt
