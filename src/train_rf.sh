#!/bin/sh

source ~/anaconda3/etc/profile.d/conda.sh
conda activate 573-project

python src/random_forest.py --rf_train_config src/configs/random_forest_training.json --train_data_path src/data/hahackathon_prepo1_train.csv --dev_data_path src/data/hahackathon_prepo1_dev.csv --results_output_path outputs/D3/random_forest/dev_classification_results.csv --param_output_path src/configs/random_forest.json
