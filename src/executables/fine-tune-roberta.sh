#!/bin/sh

source ~/anaconda3/etc/profile.d/conda.sh
conda activate /projects/assigned/2122_ling573_elibales/env/573-project/

python ../src/fine-tune-roberta.py --train_sentences ../src/data/hahackathon_prepo1_train.csv --dev_sentences ../src/data/hahackathon_prepo1_dev.csv  "$@"
