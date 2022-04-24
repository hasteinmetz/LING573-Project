conda env export --no-builds > $1 
NEWFILE=$( head -n $(expr $(cat $1 | wc -l) - 1) $1 )
echo "$NEWFILE" > $1
