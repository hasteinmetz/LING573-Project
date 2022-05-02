#!/bin/sh

python src/baseline.py --train_sentences src/data/hahackathon_prepo1_train.csv --dev_sentences src/data/hahackathon_prepo1_dev.csv --hidden_layer 100 --learning_rate 0.00001 --batch_size 64 --epochs 5 --output_file outputs/D2/d2_adam_out.txt > adam_log
