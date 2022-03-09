"""Convert job sequences to forecasting format by truncating."""
import argparse
import os
import time
import numpy as np

import pdb

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", 
                    type=str,
                    help="Directory containing sequence data.")
parser.add_argument("--dataset-name", 
                    type=str,
                    help="Name of survey dataset (e.g. 'resumes' or 'nlsy').")
args = parser.parse_args()

cutoff_year = 2014

data_dir = args.data_dir
save_dir = os.path.join(args.data_dir, "forecast_{}".format(args.dataset_name))
if not os.path.exists(save_dir):
  os.makedirs(save_dir)

"""
For each example:
 a) If all years are after the cutoff year, remove the whole example
 b) If some years are after the cutoff year but some before, remove the 
    occupations from after the cutoff year

Include cutoff_year and all jobs before it in training and validation sets.
"""

for split in ['train', 'valid']:
  print("Loading job...")
  with open(os.path.join(data_dir, "{}.job".format(split))) as temp_file:
    job = [line.rstrip('\n') for line in temp_file]
  print("...done")

  print("Loading year...")
  with open(os.path.join(data_dir, "{}.year".format(split))) as temp_file:
    year = [line.rstrip('\n') for line in temp_file]
  print("...done")
  
  print("Loading education...")
  with open(os.path.join(data_dir, "{}.education".format(split))) as temp_file:
    education = [line.rstrip('\n') for line in temp_file]
  print("...done")

  # Resume data doesn't have race/ethnicity or gender.
  if args.dataset_name != "resumes":
    print("Loading ethnicity...")
    with open(
      os.path.join(data_dir, "{}.ethnicity".format(split))) as temp_file:
      ethnicity = [line.rstrip('\n') for line in temp_file]
    print("...done")

    print("Loading gender...")
    with open(os.path.join(data_dir, "{}.gender".format(split))) as temp_file:
      gender = [line.rstrip('\n') for line in temp_file]
    print("...done")

  print("Loading location...")
  with open(os.path.join(data_dir, "{}.location".format(split))) as temp_file:
    location = [line.rstrip('\n') for line in temp_file]
  print("...done")

  job_proc = []
  year_proc = []
  education_proc = []
  gender_proc = []
  ethnicity_proc = []
  location_proc = []

  first_time = time.time()

  for ind in range(len(job)):
    if (ind + 1) % 500000 == 0:
      print("{}: Working on {}/{}, average time {:.4f}s".format(
        split.title(), ind, len(job), (time.time() - first_time) / ind))
    # Get the years
    years = [int(x) for x in year[ind].split(" ")]
    if np.all(np.array(years) > cutoff_year):
      # All years are after cutoff year, so we don't include in any split.
      continue
    elif np.any(np.array(years) >= cutoff_year):
      # Some years are after cutoff year (or last example includes it).
      # Get the indices of the years that are before or include cutoff year.
      before_inds = np.where(np.array(years) <= cutoff_year)[0]
      # Delete the offending occupations.
      new_job = " ".join([job[ind].split(" ")[x] for x in before_inds])
      new_year = " ".join([year[ind].split(" ")[x] for x in before_inds])
      new_education = " ".join(
        [education[ind].split(" ")[x] for x in before_inds])
    elif np.all(np.array(years) < cutoff_year):
      # All years are before cutoff year
      new_job = job[ind] 
      new_year = year[ind]
      new_education = education[ind]

    if args.dataset_name != "resumes":
      new_ethnicity = ethnicity[ind]
      new_gender = gender[ind]
      ethnicity_proc.append(new_ethnicity)
      gender_proc.append(new_gender)
    
    new_location = location[ind]
    
    job_proc.append(new_job)
    year_proc.append(new_year)
    education_proc.append(new_education)
    location_proc.append(new_location)

  print("Finished with {} examples kept in {}".format(len(job_proc), split))

  print("Saving job...")
  with open(os.path.join(save_dir, '{}.job'.format(split)), 'w') as f:
    for item in job_proc:
      f.write("%s\n" % item)
  print("...done")
  del job_proc

  print("Saving year...")
  with open(os.path.join(save_dir, '{}.year'.format(split)), 'w') as f:
    for item in year_proc:
      f.write("%s\n" % item)
  print("...done")
  del year_proc

  print("Saving education...")
  with open(os.path.join(save_dir, '{}.education'.format(split)), 'w') as f:
    for item in education_proc:
      f.write("%s\n" % item)
  print("...done")
  del education_proc

  if args.dataset_name != "resumes":
    print("Saving ethnicity...")
    with open(os.path.join(save_dir, '{}.ethnicity'.format(split)), 'w') as f:
      for item in ethnicity_proc:
        f.write("%s\n" % item)
    print("...done")
    del ethnicity_proc

    print("Saving gender...")
    with open(os.path.join(save_dir, '{}.gender'.format(split)), 'w') as f:
      for item in gender_proc:
        f.write("%s\n" % item)
    print("...done")
    del gender_proc

  print("Saving location...")
  with open(os.path.join(save_dir, '{}.location'.format(split)), 'w') as f:
    for item in location_proc:
      f.write("%s\n" % item)
  print("...done")
  del location_proc

# print("Creating test set.")

# split = 'test'
# # For resume data, we use the original test set as the test set (only including
# # sequences that contain at least one job before the cutoff year).
# # Otherwise, we use the whole data as the test set (again only including 
# # sequences that contain at least one job before the cutoff year).
# if args.dataset_name == "resumes":  
#   print("Loading job...")
#   with open(os.path.join(data_dir, "{}.job".format(split))) as temp_file:
#     job = [line.rstrip('\n') for line in temp_file]
#   print("...done")
  
#   print("Loading year...")
#   with open(os.path.join(data_dir, "{}.year".format(split))) as temp_file:
#     year = [line.rstrip('\n') for line in temp_file]
#   print("...done")
  
#   print("Loading location...")
#   with open(os.path.join(data_dir, "{}.location".format(split))) as temp_file:
#     location = [line.rstrip('\n') for line in temp_file]
#   print("...done")
  
#   print("Loading education...")
#   with open(os.path.join(data_dir, "{}.education".format(split))) as temp_file:
#     education = [line.rstrip('\n') for line in temp_file]
#   print("...done")
# else:
#   pdb.set_trace()


# job_proc = []
# year_proc = []
# gender_proc = []
# ethnicity_proc = []
# location_proc = []
# education_proc = []

# for ind in range(len(job)):
#   # Include all job for test set.
#   new_job = job[ind]  
#   new_year = year[ind] 
#   new_education = education[ind] 
#   new_location = location[ind]
#   if args.dataset_name != "resumes":
#     new_ethnicity = ethnicity[ind]
#     new_gender = gender[ind]
#     education_proc.append(new_education)
#     ethnicity_proc.append(new_ethnicity)
  
#   job_proc.append(new_job)
#   year_proc.append(new_year)
#   gender_proc.append(new_gender)
#   location_proc.append(new_location)
  
# print("Saving job...")
# with open(os.path.join(save_dir, '{}.job'.format(split)), 'w') as f:
#   for item in job_proc:
#     f.write("%s\n" % item)
# print("...done")

# print("Saving year...")
# with open(os.path.join(save_dir, '{}.year'.format(split)), 'w') as f:
#   for item in year_proc:
#     f.write("%s\n" % item)
# print("...done")

# print("Saving education...")
# with open(os.path.join(save_dir, '{}.education'.format(split)), 'w') as f:
#   for item in education_proc:
#     f.write("%s\n" % item)
# print("...done")

# if args.dataset_name != "resumes":
#   print("Saving ethnicity...")
#   with open(os.path.join(save_dir, '{}.ethnicity'.format(split)), 'w') as f:
#     for item in ethnicity_proc:
#       f.write("%s\n" % item)
#   print("...done")
  
#   print("Saving gender...")
#   with open(os.path.join(save_dir, '{}.gender'.format(split)), 'w') as f:
#     for item in gender_proc:
#       f.write("%s\n" % item)
#   print("...done")

# print("Saving location...")
# with open(os.path.join(save_dir, '{}.location'.format(split)), 'w') as f:
#   for item in location_proc:
#     f.write("%s\n" % item)
# print("...done")
