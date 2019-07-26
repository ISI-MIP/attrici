module load cdo/1.9.6/gnu-threadsafe
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
outputfile=${datafolder}/input/${variable}_${dataset}.nc4
inputfolder=${datafolder}/links/
echo 'Outputfile:' $outputfile
echo 'Inputfiles:'
echo ${inputfolder}${variable}_${dataset}_????_????.nc* 

if [ -e ${outputfile} ]; then
    while true; do
        read -p "Specified output file exists. Delete?" yn
        case $yn in
            [Yy'\r']* ) \
                echo 'Deleting' $outputfile
                rm $outputfile 
                echo 'Merging files!'
                cdo mergetime ${inputfolder}${variable}_${dataset}_????_????.nc* \
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
    cdo mergetime ${inputfolder}${variable}_${dataset}_????_????.nc* \
    $outputfile
fi
echo 'Done!'
