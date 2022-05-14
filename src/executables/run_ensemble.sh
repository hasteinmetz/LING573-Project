#!/bin/sh

cd /projects/assigned/2122_ling573_elibales/repo/
source ~/anaconda3/etc/profile.d/conda.sh
conda activate /projects/assigned/2122_ling573_elibales/env/

python src/ensemble_hilly.py --train_data_path src/data/hahackathon_prepo1_train.csv --dev_data_path src/data/hahackathon_prepo1_dev.csv --output_file outputs/D4/ensemble-test/ensemble.csv --hurtlex_path src/data/hurtlex_en.tsv
