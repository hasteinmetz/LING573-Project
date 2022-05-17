#!/bin/sh

echo "Saving environment.yml file. NOTE: this file is meant to create environments on Linux."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate 573-project
conda env export --from-history --channel pytorch --channel c-forge > $1 
NEWFILE=$( head -n $(expr $(cat $1 | wc -l) - 1) $1 )
echo "$NEWFILE" > $1
