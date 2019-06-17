# run from parent directory
if [ -e settings.py ]; then
    settings_file=settings.py 
else 
    settings_file=../settings.py
fi 
variable="$(grep 'variable =' ${settings_file} | cut -d' ' -f3 \
    | sed "s/'//g" | sed 's/"//g')"
echo 'Merging data for variable' $variable
dataset="$(grep 'dataset =' ${settings_file} | cut -d' ' -f3 \
    | sed "s/'//g" | sed 's/"//g')"
# startyear="$(grep 'startyear =' ${settings_file} | cut -d' ' -f3 \
#     | sed "s/'//g" | sed 's/"//g')"
# endyear="$(grep 'endyear =' ${settings_file} | cut -d' ' -f3 \
#     | sed "s/'//g" | sed 's/"//g')"
# datafolder selections relies on the folder being wrapped in double quotation marks
datafolder="$(grep 'data_dir =' ${settings_file} | grep $USER | cut -d'"' -f2 | \
    sed "s/'//g" | sed 's/"//g')"
outputfile=${datafolder}${variable}_${dataset}_gregorian.nc4
echo 'Outputfile:' $outputfile
echo 'Inputfiles:'
echo ${datafolder}${variable}_${dataset}_????_????.nc* 

if [ -e ${outputfile} ]; then
    while true; do
        read -p "Specified output file exists. Delete?" yn
        case $yn in
            [Yy'\r']* ) \
                echo 'Deleting' $outputfile
                rm $outputfile 
                echo 'Merging files!'
                cdo mergetime ${datafolder}${variable}_${dataset}_????_????.nc* \
                $outputfile
                echo 'Done with merge!'
            break;;
            [Nn]* ) \
                echo 'Aborted'
                exit;;
            * ) echo "Please answer yes or no.";;
        esac
    done
else
    echo 'Merging files!'
    cdo mergetime ${datafolder}${variable}_${dataset}_????_????.nc* \
    $outputfile
fi
echo 'Done!'
