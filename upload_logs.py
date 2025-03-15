from utils.log import NeptuneLogger
import argparse

# Get arg from command line
parser = argparse.ArgumentParser()
parser.add_argument('--dir_path', type=str, required=True)
parser.add_argument('--algorithm', type=str, required=True)
args = parser.parse_args()

# Create NeptuneLogger object
nl = NeptuneLogger(log_dir_path=args.dir_path, algorithm=args.algorithm)
nl.start()
