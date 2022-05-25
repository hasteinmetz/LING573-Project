#!/bin/sh

source ~/anaconda3/etc/profile.d/conda.sh
conda activate /projects/assigned/2122_ling573_elibales/env/

python src/pretraining-test.py --train_data_path src/data/D4_1_hahackathon_prepo1_train.csv --dev_data_path src/data/D4_1_hahackathon_prepo1_dev.csv --error_path src/data/pretrained-misclassified --model_save_path src/models/pretraining-test --output_path outputs/D4/pretraining-test --hurtlex_path src/data/hurtlex_en.tsv "$@"
