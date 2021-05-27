#/bin/bash

output_dir=/p/tmp/mengel/isimip/attrici/output/
collected_output_dir=/p/tmp/mengel/isimip/attrici/collected_output/20210527_gswp3v109-w5e5

for var in tas tasskew tasrange pr hurs sfcwind rlds rsds ps tasmin tasmax
do
  ln -s ${output_dir}/isicf035_gswp3v109-w5e5_${var}_sub01/cfact/${var}/${var}_GSWP3-W5E5_cfactual_rechunked_valid.nc4 \
  $collected_output_dir/gswp3v109-w5e5_counterclim_${var}_global_daily_1901_2019_intermediate.nc
done

