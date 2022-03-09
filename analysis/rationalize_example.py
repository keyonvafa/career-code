"""Rationalize an example job sequence."""
import argparse
import os
import pdb
import torch
import torch.nn.functional as F

from fairseq import utils
from fairseq.models.transformer import TransformerModel
from rationalization import rationalize_occupation_model
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--binary-data-dir", 
                    type=str,
                    help="Location of binarized data.")
parser.add_argument("--save-dir", 
                    type=str,
                    help="Location of saved model checkpoint (must contain a "
                         "saved model called "
                         "'career-transferred-word-dropout').")
parser.add_argument("--dataset-name", 
                    type=str,
                    help="Name of survey dataset (e.g. 'nlsy').")
args = parser.parse_args()

## Change this if you saved the model under a different name.
model_path = os.path.join(
  args.save_dir, 
  '{}/career-transferred-word-dropout'.format(args.dataset_name))
binary_data_path = os.path.join(args.binary_data_dir, args.dataset_name)

# Load model.
model = TransformerModel.from_pretrained(
  model_path,
  checkpoint_file="checkpoint_best.pt",
  data_name_or_path=binary_data_path)

# Move model to GPU and set to eval mode.
if torch.cuda.is_available():
  model.cuda()
model.eval()
model.model = model.models[0]

# Load test data.
model.task.load_dataset('test')
itr = model.task.get_batch_iterator(
  dataset=model.task.dataset('test'),
  max_tokens=1200,
  max_sentences=100,).next_epoch_itr(shuffle=False)

for eval_index, sample in enumerate(itr):
  print("Evaluating {}/{}...".format(eval_index, itr.total))
  with torch.no_grad():
    if torch.cuda.is_available():
      sample = utils.move_to_cuda(sample)
    # As an example, rationalize the first sequence in the batch.
    non_pad_ids = (
      sample['net_input']['src_tokens'][0, :] != model.task.dictionary.pad())
    input_jobs = sample['net_input']['src_tokens'][0, non_pad_ids]
    input_years = sample['net_input']['years'][0, non_pad_ids]
    input_educations = sample['net_input']['educations'][0, non_pad_ids]
    input_ethnicities = sample['net_input']['ethnicities'][0]
    input_genders = sample['net_input']['genders'][0]
    input_locations = sample['net_input']['locations'][0]
    rationale_log = rationalize_occupation_model(
      model, input_jobs, years=input_years, educations=input_educations,
      ethnicities=input_ethnicities, genders=input_genders, 
      locations=input_locations, verbose=True)
    break
