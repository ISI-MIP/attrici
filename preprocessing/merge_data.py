import subprocess
from pathlib import Path

variable_list = ["pr"]
# out of "GSWP3", "GSWP3+ERA5" etc. see source_base for more datasets.
dataset = "GSWP3"

source_base = Path(
    "/p/projects/isimip/isimip/ISIMIP2a/InputData/climate_co2/climate/HistObs/"
)

source_dir = source_base / dataset

output_base = Path("/p/tmp/mengel/isimip/isi-cfact/input/")

output_dir = output_base / dataset
output_dir.mkdir(exist_ok=True)


for variable in variable_list:

    output_file = output_dir / Path(variable + "_" + dataset.lower() + "_merged.nc4")

    cmd = (
        "module load cdo && cdo mergetime "
        + str(source_dir)
        + "/"
        + variable
        + "_"
        + dataset.lower()
        + "_????_????.nc* "
        + str(output_file)
    )
    print(cmd)
    subprocess.check_call(cmd, shell=True)
