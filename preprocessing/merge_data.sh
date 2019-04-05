# run from parent directory
if [ -e settings.py ]; then
    settings_file=settings.py 
else 
    settings_file=../settings.py
fi 
variable="$(grep 'variable =' ${settings_file} | cut -d' ' -f3)"
echo 'Merging data for variable ' $variable
dataset=gswp3
startyear=1901
endyear=2010
datapath=/p/tmp/bschmidt/${dataset}/
cdo mergetime ${datapath}${variable}_${dataset}_*.nc* \
    ${datapath}${variable}_${dataset}_${startyear}_${endyear}.nc4
