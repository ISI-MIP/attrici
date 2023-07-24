import pathlib

def count_files_in_directory(directory, file_extension):
    path = pathlib.Path(directory)
    files = path.rglob(f"*{file_extension}")
    count = sum(1 for file in files if file.is_file())
    return count

directory_path = "/p/tmp/sitreu/data/attrici/output/attrici_03_era5_t00009_hurs_rechunked/timeseries/hurs"
file_extension = ".h5"
num_files = count_files_in_directory(directory_path, file_extension)
# assert landmask.count==num_files
num_files

