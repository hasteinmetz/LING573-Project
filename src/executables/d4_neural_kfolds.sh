#!/bin/sh

source ~/anaconda3/etc/profile.d/conda.sh
conda activate /projects/assigned/2122_ling573_elibales/env/

time python src/neural_ensemble_kfolds.py --train_data_path src/data/D4_hahackathon_prepo1_train.csv --test_data_path src/data/D4_hahackathon_prepo1_ --hurtlex_path src/data/hurtlex_en.tsv --error_path src/data/nn-kfolds-misclassified --model_save_path src/models/ --output_path outputs/D4/ "$@"

cp outputs/D4/primary/devtest/D4_scores.out results/D4/primary/devtest/D4_scores.out
cp outputs/D4/primary/evaltest/D4_scores.out results/D4/primary/evaltest/D4_scores.out
cp outputs/D4/adaptation/devtest/D4_scores.out results/D4/adaptation/devtest/D4_scores.out
cp outputs/D4/adaptation/evaltest/D4_scores.out results/D4/adaptation/evaltest/D4_scores.out