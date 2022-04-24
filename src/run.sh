#!/bin/sh

python src/baseline.py --train_sentences src/data/hahackathon_prepo1_dev.csv --dev_sentences src/data/hahackathon_prepo1_dev.csv --hidden_layer 100 --learning_rate 0.0001 --batch_size 100 --epochs 64 --random_seeds src/data/random_seeds_100.txt --output_file outputs/baseline_output.csv
