#!/bin/sh

python src/baseline.py --train_sentences src/data/hahackathon_prepo1_train.csv --dev_sentences src/data/hahackathon_prepo1_dev.csv --hidden_layer 100 --learning_rate 0.0001 --batch_size 64 --epochs 1 --output_file outputs/D2/d2_out.txt
