"""Modify year dictionary so that years are ordered."""
import argparse
import os

from collections import Counter

import pdb
import time

start = time.time()
parser = argparse.ArgumentParser()
parser.add_argument("--data-dir",
                    type=str,
                    help="Directory containing raw data.")
args = parser.parse_args()

data_dir = args.data_dir
save_dir = args.data_dir

with open(os.path.join(data_dir, "dict.txt")) as temp_file:
  year_dict = [line.rstrip('\n') for line in temp_file]

years = []
counts = []

for i in range(len(year_dict)):
  years.append(year_dict[i].split(" ")[0])
  counts.append(year_dict[i].split(" ")[1])

new_year_dict = []

sorted_years = sorted(years)
for i in range(len(year_dict)):
  new_year_dict.append(sorted_years[i] + " " + counts[i])

with open(os.path.join(save_dir, 'dict.txt'), 'w') as f:
  for item in new_year_dict:
    f.write("%s\n" % item)