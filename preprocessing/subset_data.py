import subprocess
from pathlib import Path

variable_list = ["tasmin","tasmax"]
# out of "GSWP3", "GSWP3+ERA5" etc. see source_base for more datasets.
dataset = "GSWP3"
sub = 5

output_base = Path("/p/tmp/mengel/isimip/isi-cfact/input/")
output_dir = output_base / dataset


for variable in variable_list:

    input_file = output_dir / Path(variable + "_" + dataset.lower() + "_merged.nc4")

    cmd = (
        "module load cdo && cdo samplegrid,"
        + str(sub)
        + " "
        + str(input_file)
        + " "
        + str(input_file).replace("_merged.nc4", "_sub"+str(sub)+".nc4")
    )
    print(cmd)
    subprocess.check_call(cmd, shell=True)
