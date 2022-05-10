#!/bin/sh

source ~/anaconda3/etc/profile.d/conda.sh
conda activate /projects/assigned/2122_ling573_elibales/env/573-project

time python ../src/lexical_features.py ../src/data/hahackathon_prepo1_train.csv ../src/data/hahackathon_prepo1_dev.csv
