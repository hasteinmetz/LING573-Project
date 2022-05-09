#!/bin/sh

source ~/anaconda3/etc/profile.d/conda.sh
conda activate 573-project

python src/ensemble.py --train_data_path src/data/hahackathon_prepo1_train.csv --dev_data_path src/data/hahackathon_prepo1_dev.csv --output_file outputs/D3/ensemble/basic_test.out