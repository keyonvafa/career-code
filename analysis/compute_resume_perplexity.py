"""Compute the test perplexity of each model on the resume data."""
import argparse
import os
import pdb
import torch
import torch.nn.functional as F
from fairseq import utils

from fairseq.models.transformer import TransformerModel
# from fairseq.models.bag_of_jobs import BagOfJobsModel
# from fairseq.models.lstm import LSTMModel
# from fairseq.models.regression import RegressionModel
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--binary-data-dir", 
                    type=str,
                    help="Location of binarized data.")
parser.add_argument("--save-dir", 
                    type=str,
                    help="Location of saved model checkpoint.")
parser.add_argument("--model-name", 
                    type=str,
                    default='career',
                    help="Model name (must be one of: 'career', 'bag-of-jobs',"
                         " 'regression', or 'lstm').")
parser.add_argument('--no-covariates', 
                    action='store_true',
                    help="Load model that does not use covariates.")
args = parser.parse_args()

model_path = os.path.join(
  args.save_dir, 
  'resumes/{}{}'.format(
    args.model_name, '_no_covariates' if args.no_covariates else ''))
binary_data_path = os.path.join(args.binary_data_dir, "resumes")

# Load model.
if args.model_name == 'career':
  model = TransformerModel.from_pretrained(
    model_path,
    checkpoint_file="checkpoint_best.pt",
    data_name_or_path=binary_data_path)

# Move model to GPU and set to eval mode.
if torch.cuda.is_available():
  model.cuda()
model.eval()
model.model = model.models[0]
two_stage = model.model.decoder.args.two_stage

# Load test data.
model.task.load_dataset('test')
itr = model.task.get_batch_iterator(
  dataset=model.task.dataset('test'),
  max_tokens=1200,
  max_sentences=100,).next_epoch_itr(shuffle=False)

all_nll = []
new_token_nll = []
non_consecutive_repeat_nll = []
consecutive_repeat_nll = []

for eval_index, sample in enumerate(itr):
  print("Evaluating {}/{}...".format(eval_index, itr.total))
  with torch.no_grad():
    if torch.cuda.is_available():
      sample = utils.move_to_cuda(sample)
    
    # Compute log probs for the sample.
    sample['net_input']['prev_output_tokens'] = sample[
      'net_input']['src_tokens']
    del sample['net_input']['src_tokens']
    output = model.model.decoder(**sample['net_input'])
    lprobs = model.model.get_normalized_probs(
      output, log_probs=True, two_stage=two_stage, 
      prev_tokens=sample['net_input']['prev_output_tokens'])
    batch_size, seq_len, _ = lprobs.size()
    nll_loss = F.nll_loss(
      lprobs.transpose(2, 1), sample['target'].to(model.device), 
      ignore_index=model.model.decoder.padding_idx, reduction="none",)
    
    # Decompose loss.
    for batch_ind in range(batch_size):
      for i in range(len(sample['target'][batch_ind])):
        # Include all non-pad and non-eos tokens in perplexity calculation.
        token_to_predict = sample['target'][batch_ind][i].item()
        if (token_to_predict != model.task.dictionary.pad_index and 
            token_to_predict != model.task.dictionary.eos_index):
          all_nll.append(nll_loss[batch_ind][i].item())
          # Don't include first job in loss decomposition.
          if i > 0:
            if token_to_predict not in sample['target'][batch_ind][:i]:
              new_token_nll.append(nll_loss[batch_ind][i].item())
            elif token_to_predict != sample['target'][batch_ind][i - 1]:
              non_consecutive_repeat_nll.append(nll_loss[batch_ind][i].item())
            else:
              consecutive_repeat_nll.append(nll_loss[batch_ind][i].item())

overall_perplexity = np.exp(np.mean(all_nll))
new_token_perplexity = np.exp(np.mean(new_token_nll))
non_consecutive_repeat_perplexity = np.exp(np.mean(non_consecutive_repeat_nll))
consecutive_repeat_perplexity = np.exp(np.mean(consecutive_repeat_nll))

print("..................")
print("Test-set results for {}{}, model loaded from '{}'".format(
  args.model_name, " (no_covariates)" if args.no_covariates else "",
  model_path))
print("..................")

print("Overall perplexity: {:.2f}".format(overall_perplexity))
print("New token perplexity: {:.2f}".format(new_token_perplexity))
print("Non-consecutive repeat perplexity: {:.2f}".format(
  non_consecutive_repeat_perplexity))
print("Consecutive repeat perplexity: {:.2f}".format(
  consecutive_repeat_perplexity))
print("..................")
