#!/bin/sh

source ~/anaconda3/etc/profile.d/conda.sh
conda activate /projects/assigned/2122_ling573_elibales/env/

## primary 
# eval data
python ../src/rf_ensemble.py --roberta_config ../src/configs/fine_tuned_roberta.json --random_forest_config ../src/configs/random_forest.json --logistic_regression_config ../src/configs/logistic_regression.json --dim_reduc_method pca --train_data_path ../src/data/hahackathon_prepo1_train.csv --dev_data_path ../src/data/hahackathon_prepo1_test.csv --output_file ../outputs/D4/rf_ensemble/d4_rfens_pca_pritest_out.txt --results_file ../results/D4_rfens_pca_pritest_scores.out --hurtlex_path /projects/assigned/2122_ling573_elibales/repo/src/data/hurtlex_en.tsv
