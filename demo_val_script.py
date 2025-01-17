import time
import argparse
import csv

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument( '--temp', type=float, default=0.3)
    parser.add_argument( '--p', type=float, default=0.9)
    return parser.parse_args()

def main():
    print('Hello, World!')
    args = options()
    print(f"Running with temp={args.temp} and p_value={args.p}")
    time.sleep(10)


    # Create mock data
    psuccess = 0.043
    success = 0.57

    # Define the CSV file path
    csv_file_path = 'output.csv'

    # Write data to CSV file
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([args.temp, args.p, psuccess, success])
    print(f'Run complete with temp={args.temp} and p_value={args.p}')



if __name__ == '__main__':
    main()
