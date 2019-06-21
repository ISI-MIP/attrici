
# Minimal script for extracting a regularly spaced subset from a larger dataset

sub=5

ncks -d lon,,,$sub -d lat,,,$sub $1 $2
