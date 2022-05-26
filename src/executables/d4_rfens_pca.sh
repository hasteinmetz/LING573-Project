#!/bin/sh

source ~/anaconda3/etc/profile.d/conda.sh
conda activate /projects/assigned/2122_ling573_elibales/env/

## primary 
# dev
python ../src/rf_ensemble.py --roberta_config ../src/configs/fine_tuned_roberta.json --random_forest_config ../src/configs/random_forest.json --logistic_regression_config ../src/configs/logistic_regression.json --dim_reduc_method pca --train_data_path ../src/data/hahackathon_prepo1_train.csv --dev_data_path ../src/data/hahackathon_prepo1_dev.csv --output_file ../outputs/D4/rf_ensemble/d4_rfens_pca_pridev_out.txt --results_file ../results/D4_rfens_pca_pridev_scores.out --hurtlex_path /projects/assigned/2122_ling573_elibales/repo/src/data/hurtlex_en.tsv

# eval data
python ../src/rf_ensemble.py --roberta_config ../src/configs/fine_tuned_roberta.json --random_forest_config ../src/configs/random_forest.json --logistic_regression_config ../src/configs/logistic_regression.json --dim_reduc_method pca --train_data_path ../src/data/hahackathon_prepo1_train.csv --dev_data_path ../src/data/hahackathon_prepo1_test.csv --output_file ../outputs/D4/rf_ensemble/d4_rfens_pca_pritest_out.txt --results_file ../results/D4_rfens_pca_pritest_scores.out --hurtlex_path /projects/assigned/2122_ling573_elibales/repo/src/data/hurtlex_en.tsv

## adaptation
# dev
python ../src/rf_ensemble.py --roberta_config ../src/configs/fine_tuned_roberta.json --random_forest_config ../src/configs/random_forest.json --logistic_regression_config ../src/configs/logistic_regression.json --dim_reduc_method pca --train_data_path ../src/data/D4_hahackathon_prepo1_train.csv --dev_data_path ../src/data/D4_hahackathon_prepo1_dev.csv --output_file ../outputs/D4/rf_ensemble/d4_rfens_pca_adadev_out.txt --results_file ../results/D4_rfens_pca_adadev_scores.out --hurtlex_path /projects/assigned/2122_ling573_elibales/repo/src/data/hurtlex_en.tsv

# eval data
python ../src/rf_ensemble.py --roberta_config ../src/configs/fine_tuned_roberta.json --random_forest_config ../src/configs/random_forest.json --logistic_regression_config ../src/configs/logistic_regression.json --dim_reduc_method pca --train_data_path ../src/data/D4_hahackathon_prepo1_train.csv --dev_data_path ../src/data/D4_hahackathon_prepo1_test.csv --output_file ../outputs/D4/rf_ensemble/d4_rfens_pca_adatest_out.txt --results_file ../results/D4_rfens_pca_adatest_scores.out --hurtlex_path /projects/assigned/2122_ling573_elibales/repo/src/data/hurtlex_en.tsv
