# Run complete preprocessing routine

bash merge_data.sh
bash remove_leap_days.sh
bash rechunk.sh
python3 create_test_data.py
python3 calc_gmt_by_ssa.py
