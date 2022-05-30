source ~/anaconda3/etc/profile.d/conda.sh
conda env create --force -f /projects/assigned/2122_ling573_elibales/repo/src/configs/environment.yml --prefix /projects/assigned/2122_ling573_elibales/env
conda activate /projects/assigned/2122_ling573_elibales/env
python3 -m spacy download en_core_web_sm