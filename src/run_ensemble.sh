#!/bin/sh

source ~/anaconda3/etc/profile.d/conda.sh
conda activate 573-project

python src/ensemble.py --random_forest_config src/configs/random_forest.json --logistic_regression_config src/configs/logistic_regression.json --train_data_path src/data/hahackathon_prepo1_train.csv --dev_data_path src/data/hahackathon_prepo1_dev.csv --output_file outputs/ensemble/d3_out.txt 
