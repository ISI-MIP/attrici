import subprocess
from pathlib import Path

""" This code allows to create tasmax and tasmin from tas, tasrange and
    tasskew. It runs independent from the attrici package.
    Copy anywhere and adjust the variables below. """

output_base = Path("/p/tmp/mengel/isimip/attrici/output/")
dataset = "GSWP3-W5E5"

tas_runid = "isicf035_gswp3v109-w5e5_tas_sub01"
tasskew_runid = "isicf035_gswp3v109-w5e5_tasskew_sub01"
tasrange_runid = "isicf035_gswp3v109-w5e5_tasrange_sub01"
out_runid = "isicf035_gswp3v109-w5e5_"
def d(var):
    """ get file strings from cfact calculation """
    tdict = {"tas":tas_runid, "tasrange":tasrange_runid, "tasskew":tasskew_runid}
    dpath = output_base/tdict[var]/"cfact"/var
    return str(dpath)+"/"+var+"_"+dataset+"_cfactual_rechunked_valid.nc4 "

def dout(var):
    """ get file strings for output data """
    dpath = output_base/f"{out_runid}{var}_sub01"/"cfact"/var
    dpath.mkdir(parents=True,exist_ok=True)
    return str(dpath)+"/"+var+"_"+dataset+"_cfactual_rechunked_valid.nc4 "

p = "module load cdo && cdo -O "

cmd = p+"-chname,tas,tasmin -chname,tas_orig,tasmin_orig -sub "+d("tas")+" -mul "+d("tasskew")+d("tasrange")+dout("tasmin")
print(cmd)
print("")
subprocess.check_call(cmd, shell=True)
cmd = p+"-chname,tasmin,tasmax -chname,tasmin_orig,tasmax_orig -add "+dout("tasmin")+d("tasrange")+dout("tasmax")
print(cmd)
subprocess.check_call(cmd, shell=True)
