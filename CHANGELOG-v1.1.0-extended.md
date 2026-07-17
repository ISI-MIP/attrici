# Changelog: legacy/v1.1.0-extended vs v1.1.0

## Added

### calibration_start and calibration_stop configuration
- Renamed `startdate` → `calibration_start`; added `calibration_stop` in `settings.py`
- Calibration period can be restricted to [calibration_start, calibration_stop]
- Model is fit on this period; detrending is applied to the full dataset

### GMT normalization from calibration period
- GMT min/max for scaling are now derived only from the calibration period
- Supports calibration_start only, calibration_stop only, or both
- Ensures normalization is consistent when applying to different application periods

### y and t normalization from calibration period
- `t_scaled` is anchored to calibration period bounds so overlap is invariant when `application_end` extends
- Variable scaling (`scale_to_unity`, `scale_and_mask`, `scale_precip`) derives min/max or gamma fit from calibration subset only, then applies to full series
- Fixed-bound variables (`hurs`, `rsds`, `tasskew`, `prsnratio`) unchanged

### Time range alignment validation
- GMT and input dataset must cover the same time range
- Validation ensures GMT start/end are within tolerance of input start/end
- Tolerance for start/end misalignment is derived from the GMT file's equidistant interval (e.g. 10 days if GMT is every 10th day)
- Checks: input_min - tolerance < gmt_min < input_min + tolerance, same for max

### Legacy Singularity container
- `containers/legacy/v1.1.0-extended/attrici-v1.1.0-extended.def` clones the extended branch from GitHub

## Modified

- `settings.py`: Renamed `startdate` → `calibration_start`; added `calibration_stop`
- `attrici/datahandler.py`: `get_subset` supports calibration_start/calibration_stop; `create_dataframe` uses calibration-period GMT, `t`, and `y` scaling; new `validate_time_range_alignment`
- `attrici/const.py`: scaling helpers accept optional `calibration_mask` for calibration-anchored min/max and precip gamma fit
- `attrici/estimator.py`: Passes `calibration_start` and `calibration_stop` to `get_subset`
- `run_estimation.py`: Passes calibration_start/calibration_stop to `create_dataframe`; loads GMT time; calls time alignment validation
- `run_single_cell.py`: Same updates as run_estimation.py for consistency
