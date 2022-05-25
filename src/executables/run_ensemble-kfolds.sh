#!/bin/sh

source ~/anaconda3/etc/profile.d/conda.sh
conda activate /projects/assigned/2122_ling573_elibales/env/

time python src/ensemble_hilly_kfolds.py --train_data_path src/data/D4_hahackathon_prepo1_train.csv --dev_data_path src/data/D4_hahackathon_prepo1_dev.csv --hurtlex_path src/data/hurtlex_en.tsv --error_path src/data/ensemble-misclassified --model_save_path src/models/testing-ensemble --output_path outputs/D4/ensemble-test "$@"
