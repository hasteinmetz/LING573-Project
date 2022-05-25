#!/bin/sh

source ~/anaconda3/etc/profile.d/conda.sh
conda activate /projects/assigned/2122_ling573_elibales/env/

python ../src/ensemble_apai.py --roberta_config ../src/configs/fine_tuned_roberta.json --random_forest_config ../src/configs/random_forest.json --logistic_regression_config ../src/configs/logistic_regression.json --dim_reduc_method mutual_info --train_data_path ../src/data/D4_hahackathon_prepo1_train.csv --dev_data_path ../src/data/D4_hahackathon_prepo1_dev.csv --output_file ../outputs/D4/rf_ensemble/d4_rfens_featsel_out.txt --results_file ../results/D4_rfens_featsel_scores.out --hurtlex_path /projects/assigned/2122_ling573_elibales/repo/src/data/hurtlex_en.tsv
