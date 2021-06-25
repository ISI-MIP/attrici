#!/bin/bash

data_dir=/p/tmp/mengel/isimip/attrici/collected_output/20210527_gswp3v109-w5e5/decadal_data
data_dir=/p/tmp/buechner/temp/attrici

for input_file in ${data_dir}/*.nc;do
  echo $input_file
  ncatted -h \
  -a history,global,d,, \
  -a source,global,d,, \
  -a CDO,global,d,, \
  -a CDI,global,d,, \
  -a comment,global,d,, \
  -a comment_PIK,global,d,, \
  -a cfact_version,global,d,, \
  -a runid,global,d,, \
  -a NCO,global,d,, \
  -a title,global,o,c,"ISIMIP3a counterfactual GSWP3-W5E5 climate" \
  -a institution,global,o,c,"Potsdam Institute for Climate Impact Research (PIK)" \
  -a project,global,o,c,"Inter-Sectoral Impact Model Intercomparison Project phase 3a (ISIMIP3a)" \
  -a contact,global,o,c,"ISIMIP cross-sectoral science team <info@isimip.org> <https://www.isimip.org>" \
  -a summary,global,o,c,"A counterfactual climate for impact attribution derived from GSWP3-W5E5. Created with ATTRICI v1.1.0, see <https://github.com/ISI-MIP/attrici/releases/tag/v1.1.0>" \
  -a version,global,o,c,"This version of the counterfactual GSWP3-W5E5 climate is based on the ATTRICI version v1.1.0. GSWP3 v1.09 and W5E5 v2.0 have been used to compile GSWP3-W5E5" \
  ${input_file}
done
