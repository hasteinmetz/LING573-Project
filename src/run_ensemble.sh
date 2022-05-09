#!/bin/sh

source ~/anaconda3/etc/profile.d/conda.sh
conda activate /projects/assigned/2122_ling573_elibales/env/573-project

<<<<<<< HEAD
time python src/ensemble.py --train_data_path src/data/hahackathon_prepo1_train.csv --dev_data_path src/data/hahackathon_prepo1_dev.csv --output_file outputs/D3/ensemble/ensemble_network2.csv
=======
python src/ensemble.py --roberta_config src/configs/fine_tuned_roberta.json --random_forest_config src/configs/random_forest.json --logistic_regression_config src/configs/logistic_regression.json --train_data_path src/data/hahackathon_prepo1_train.csv --dev_data_path src/data/hahackathon_prepo1_dev.csv --output_file outputs/D3/ensemble/d3_out.txt --results_file results/D3_scores.out
>>>>>>> 731cd07 (cleaning up messy merge)
