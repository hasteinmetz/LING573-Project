#!/bin/sh

source ~/anaconda3/etc/profile.d/conda.sh
conda activate 573-project

python src/fine-tune.py --train_sentences src/data/hahackathon_prepo1_train.csv --dev_sentences src/data/hahackathon_prepo1_dev.csv --learning_rate 5e-5 --batch_size 32 --epochs 1 --output_file outputs/D3/roberta/fine-tune_results.csv --save_file src/models/roberta-fine-tuned-preproc "$@"
