if [ -e settings.py ]; then
    settings_file=settings.py 
else 
    settings_file=../settings.py
fi 
var="$(grep 'variable =' ${settings_file} | cut -d' ' -f3 \
    | sed "s/'//g")"
echo 'splitting data for variable' $variable
dataset=gswp3
startyear=1901
endyear=2010
datapath=/p/tmp/bschmidt/${dataset}/

var='tas'
dataset='gswp3'
cd $datapath
# infile=${var}_gswp3_1901_2010_noleap.nc4
infile=${var}_detrended.nc4
cdo splitsel,3650  ${infile} ${var}_detrended

mv ${var}_detrended000000.nc ${var}_detrended_${dataset}_1901_1910.nc4
mv ${var}_detrended000001.nc ${var}_detrended_${dataset}_1911_1920.nc4
mv ${var}_detrended000002.nc ${var}_detrended_${dataset}_1921_1930.nc4
mv ${var}_detrended000003.nc ${var}_detrended_${dataset}_1931_1940.nc4
mv ${var}_detrended000004.nc ${var}_detrended_${dataset}_1941_1950.nc4
mv ${var}_detrended000005.nc ${var}_detrended_${dataset}_1951_1960.nc4
mv ${var}_detrended000006.nc ${var}_detrended_${dataset}_1961_1970.nc4
mv ${var}_detrended000007.nc ${var}_detrended_${dataset}_1971_1980.nc4
mv ${var}_detrended000008.nc ${var}_detrended_${dataset}_1981_1990.nc4
mv ${var}_detrended000009.nc ${var}_detrended_${dataset}_1991_2000.nc4
mv ${var}_detrended000010.nc ${var}_detrended_${dataset}_2001_2010.nc4
