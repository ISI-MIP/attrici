import argparse
import re

def sum_numbers_from_log(log_text):
    inf_pattern = r"replace (\d+) inf values"
    nan_pattern = r"(\d+) nan values"
    logp_pattern = r"(\d+) values with too small logp \(<(-?\d+)\)"

    inf_values_sum = sum(int(match) for match in re.findall(inf_pattern, log_text))
    nan_values_sum = sum(int(match) for match in re.findall(nan_pattern, log_text))
    logp_values_sum = sum(int(count) for count, _ in re.findall(logp_pattern, log_text))
    return inf_values_sum, nan_values_sum, logp_values_sum

def main():
    parser = argparse.ArgumentParser(description="Sum the numbers from the log file.")
    parser.add_argument("file_path", type=str, help="Path to the log file.")
    args = parser.parse_args()

    # Read the log file
    with open(args.file_path, "r") as file:
        log_text = file.read()

    inf_sum, nan_sum, logp_sum = sum_numbers_from_log(log_text)
    print("Sum of inf values:", inf_sum)
    print("Sum of nan values:", nan_sum)
    print("Sum of values with too small logp (<-300):", logp_sum)

if __name__ == "__main__":
    main()

