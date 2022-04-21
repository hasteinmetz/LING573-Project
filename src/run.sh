#!/bin/sh

/Users/hillel/Documents/UW/sp2022/systems/repo/env/bin/python /Users/hillel/Documents/UW/sp2022/systems/repo/src/classifier_layer.py --raw_data src/data/hahackathon_prepo1_train.csv --input_embeddings src/embeddings/train_cpu.pt --hidden_layer 10000 --learning_rate 0.0001 --batch_size 100 --epochs 100