#!/bin/sh

source ~/anaconda3/etc/profile.d/conda.sh
conda activate /projects/assigned/2122_ling573_elibales/env/

time python src/neural_ensemble.py --train_data_path src/data/D4_hahackathon_prepo1_train.csv --test_data_path src/data/D4_hahackathon_prepo1_ --hurtlex_path src/data/hurtlex_en.tsv --error_path src/data/nn-misclassified --model_save_path src/models/neural_ensemble --output_path outputs/D4/neural_ensemble "$@"
