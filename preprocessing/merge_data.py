import subprocess
from pathlib import Path

variable_list = ["tas", "tasmax", "tasmin", "pr", "ps", "sfcwind", "rsds", "rlds", "hurs", "huss"]
# out of "GSWP3", "GSWP3+ERA5" etc. see source_base for more datasets.
dataset = "GSWP3-W5E5"

source_base = Path(
    "/p/projects/isimip/isimip/ISIMIP3a/InputData/climate/atmosphere/obsclim"
)

source_dir = source_base / dataset

output_base = Path("/p/tmp/sitreu/isimip/isi-cfact/input/")

output_dir = output_base / dataset
output_dir.mkdir(exist_ok=True)


for variable in variable_list:

    output_file = output_dir / Path(variable + "_" + dataset.lower() + "_merged.nc4")

    cmd = (
        "module load cdo && cdo mergetime "
        + str(source_dir)
        + "/"
        + dataset.lower()
        + "_obsclim_"
        + variable
        + "_global_daily"
        + "_????_????.nc* "
        + str(output_file)
    )
    print(cmd)
    subprocess.check_call(cmd, shell=True)
