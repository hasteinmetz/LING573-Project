#!/bin/sh

source ~/anaconda3/etc/profile.d/conda.sh
conda activate 573-project
conda env export --no-builds --from-history > $1 
NEWFILE=$( head -n $(expr $(cat $1 | wc -l) - 1) $1 )
echo "$NEWFILE" > $1
