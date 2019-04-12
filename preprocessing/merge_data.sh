# run from parent directory
if [ -e settings.py ]; then
    settings_file=settings.py 
else 
    settings_file=../settings.py
fi 
variable="$(grep 'variable =' ${settings_file} | cut -d' ' -f3 \
    | sed "s/'//g")"
echo 'Merging data for variable' $variable
dataset=gswp3
startyear=1901
endyear=2010
datapath=/p/tmp/bschmidt/${dataset}/
outputfile=${datapath}${variable}_${dataset}_${startyear}_${endyear}.nc4

if [ -e ${outputfile} ]; then
    while true; do
        read -p "Specified output file exists. Delete?" yn
        case $yn in
            [Yy'\r']* ) \
                echo 'Deleting' $outputfile
                rm $outputfile 
                echo 'Merging files!'
                cdo mergetime ${datapath}${variable}_${dataset}_*.nc* \
                $outputfile
            break;;
            [Nn]* ) \
                echo 'Aborted'
                exit;;
            * ) echo "Please answer yes or no.";;
        esac
    done
fi

