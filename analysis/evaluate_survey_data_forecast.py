"""Evaluate the forecasting performance of CAREER/baselines on survey data."""
import argparse
import os
import pdb
import torch
import torch.nn.functional as F
import numpy as np

from fairseq.models.transformer import TransformerModel
from fairseq.models.bag_of_jobs import BagOfJobsModel
from fairseq.models.regression import RegressionModel
from fairseq import utils
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument("--binary-data-dir", 
                    type=str,
                    help="Location of binarized data.")
parser.add_argument("--save-dir", 
                    type=str,
                    help="Location of saved model checkpoints.")
parser.add_argument("--model-name", 
                    type=str,
                    default='career',
                    help="Model name (must be one of: 'career', 'bag-of-jobs',"
                         " or 'regression').")
parser.add_argument("--dataset-name", 
                    type=str,
                    help="Name of survey dataset (e.g. 'nlsy').")
parser.add_argument("--num-samples", 
                    type=int,
                    default=1000,
                    help="Number of Monte Carlo samples.")
args = parser.parse_args()

model_path = os.path.join(
  args.save_dir, 
  'forecast-{}/{}{}'.format(
    args.dataset_name, args.model_name, 
    "-transferred" if args.model_name == 'career' else ''))
binary_data_path = os.path.join(
  args.binary_data_dir, 'forecast-{}'.format(args.dataset_name))

# Load model.
if args.model_name == 'career':
  model = TransformerModel.from_pretrained(
    model_path,
    checkpoint_file="checkpoint_best.pt",
    data_name_or_path=binary_data_path)
elif args.model_name == 'bag-of-jobs':
  model = BagOfJobsModel.from_pretrained(
    model_path,
    checkpoint_file="checkpoint_best.pt",
    data_name_or_path=binary_data_path)
elif args.model_name == 'regression':
  model = RegressionModel.from_pretrained(
    model_path,
    checkpoint_file="checkpoint_best.pt",
    data_name_or_path=binary_data_path)
else:
  raise ValueError("Model name must be one of: 'career', 'bag-of-jobs',"
                   " or 'regression'.")

if torch.cuda.is_available():
  model.cuda()
model.eval()
model.model = model.models[0]
two_stage = model.model.decoder.args.two_stage

pdb.set_trace()
sum(p.numel() for p in model.parameters())

# Load test data.
model.task.load_dataset('test')
itr = model.task.get_batch_iterator(
  dataset=model.task.dataset('test'),
  max_tokens=1200,
  max_sentences=100,).next_epoch_itr(shuffle=False)

cutoff_year = 2014  # last year to be included in the training data.
cutoff_index = model.task._year_dictionary.index(str(cutoff_year))

if args.model_name in ['career', 'regression']:
  year_weights = model.model.decoder.embed_year.weight.detach().cpu().numpy()
  last_year_weights = year_weights[cutoff_index, :]
  year_weights[cutoff_index + 1, :] = last_year_weights
  model.model.decoder.embed_year.weight.data = torch.from_numpy(
    year_weights).cuda()
else:
  current_year_weights = (
    model.model.decoder.embed_current_year.weight.detach().cpu().numpy())
  last_year_weights = current_year_weights[cutoff_index, :]
  current_year_weights[cutoff_index + 1, :] = last_year_weights
  model.model.decoder.embed_current_year.weight.data = (
    torch.from_numpy(current_year_weights).cuda())
  context_year_weights = (
    model.model.decoder.embed_context_year.weight.detach().cpu().numpy())
  last_year_weights = context_year_weights[cutoff_index, :]
  context_year_weights[cutoff_index + 1, :] = last_year_weights
  model.model.decoder.embed_context_year.weight.data = (
    torch.from_numpy(context_year_weights).cuda())

num_samples = args.num_samples
max_possible_year = max(
  [int(x) for x in model.task._year_dictionary.symbols if x.isdigit()])

all_nlls = []
unemployment_nlls = []
out_of_labor_force_nlls = []
education_nlls = []

unemployment_outcomes = []
unemployment_probs = []
out_of_labor_force_outcomes = []
out_of_labor_force_probs = []
education_outcomes = []
education_probs = []

unemployment_index = model.task.dictionary.index("unemployed")
out_of_labor_force_index = model.task.dictionary.index("not_in_labor_force")
education_index = model.task.dictionary.index("education")

for eval_index, sample in enumerate(itr):
  print("Working on {}/{}...".format(eval_index, itr.total))
  with torch.no_grad():
    if torch.cuda.is_available():
      sample = utils.move_to_cuda(sample)
    sample['net_input']['prev_output_tokens'] = sample['net_input']['src_tokens']
    del sample['net_input']['src_tokens']
    # Need to simulate all outcomes beyond 2015.
    batch_size = sample['target'].size(0)
    # Sample trajectories for each individual in the batch.
    for batch_ind in range(batch_size):
      all_years = np.array(
        [model.task._year_dictionary.string([x]) 
         for x in sample['net_input']['years'][batch_ind]])
      all_years = np.array(
        [int(x) for x in all_years if len(x) > 0 and x != '<pad>'])
      # Only forecast for individuals who start before the cutoff and have
      # observations through the maximum possible year.
      if max(all_years) == max_possible_year and min(all_years) <= cutoff_year:
        first_simulated_index = np.where(all_years > cutoff_year)[0][0]
        # Copy all pre-cutoff information for each sample to predict the first 
        # post-cutoff job.
        prev_tokens = sample['net_input']['prev_output_tokens'][batch_ind][
          :first_simulated_index + 1][None].repeat([num_samples, 1])
        years = sample['net_input']['years'][batch_ind][
          :first_simulated_index + 1][None].repeat([num_samples, 1])
        educations = sample['net_input']['educations'][batch_ind][
          :first_simulated_index + 1][None].repeat([num_samples, 1])
        ethnicities = sample['net_input']['ethnicities'][batch_ind].repeat(
          [num_samples, 1])
        genders = sample['net_input']['genders'][batch_ind].repeat(
          [num_samples, 1])
        locations = sample['net_input']['locations'][batch_ind].repeat(
          [num_samples, 1])
        # Simulate each year, one-by-one.
        num_unseen_years = sum(all_years >= cutoff_year)
        for simulated_year in range(first_simulated_index, 
                                    first_simulated_index + num_unseen_years):
          true_job = sample['target'][batch_ind][simulated_year].item()
          output = model.model.decoder.forward(
            prev_tokens, years=years, educations=educations, genders=genders, 
            ethnicities=ethnicities, locations=locations)
          probs = model.model.get_normalized_probs(
            output, log_probs=False, two_stage=True, 
            prev_tokens=prev_tokens)[:, -1]
          # Sample next job from distribution.
          next_job_samples = torch.multinomial(probs, num_samples=1)
          current_year = all_years[simulated_year]
          average_prob = probs.mean(0)
          all_nlls.append(-np.log(average_prob[true_job].item()))
          unemployment_probs.append(average_prob[unemployment_index].item())
          out_of_labor_force_probs.append(
            average_prob[out_of_labor_force_index].item())
          education_probs.append(average_prob[education_index].item())
          if true_job == unemployment_index:
            unemployment_nlls.append(
              -np.log(average_prob[unemployment_index].item()))
            unemployment_outcomes.append(1)
          else:
            unemployment_nlls.append(
              -np.log(1. - average_prob[unemployment_index].item()))
            unemployment_outcomes.append(0)
          if true_job == out_of_labor_force_index:
            out_of_labor_force_nlls.append(
              -np.log(average_prob[out_of_labor_force_index].item()))
            out_of_labor_force_outcomes.append(1)
          else:
            out_of_labor_force_nlls.append(
              -np.log(1. - average_prob[out_of_labor_force_index].item()))
            out_of_labor_force_outcomes.append(0)
          if true_job == education_index:
            education_nlls.append(
              -np.log(average_prob[education_index].item()))
            education_outcomes.append(1)
          else:
            education_nlls.append(
              -np.log(1. - average_prob[education_index].item()))
            education_outcomes.append(0)
          # Update for next year.
          prev_tokens = torch.cat([prev_tokens, next_job_samples], -1)
          years = sample['net_input']['years'][batch_ind][
            :(simulated_year + 2)][None].repeat([num_samples, 1])
          educations = sample['net_input']['educations'][batch_ind][
            :(simulated_year + 2)][None].repeat([num_samples, 1])

overall_ppl = np.exp(np.mean(all_nlls))
unemployment_ppl = np.exp(np.mean(unemployment_nlls))
out_of_labor_force_ppl = np.exp(np.mean(out_of_labor_force_nlls))
education_ppl = np.exp(np.mean(education_nlls))

unemployment_auc = roc_auc_score(unemployment_outcomes, unemployment_probs)
out_of_labor_force_auc = roc_auc_score(
  out_of_labor_force_outcomes, out_of_labor_force_probs)
education_auc = roc_auc_score(education_outcomes, education_probs)

print("....................................................")
print("Overall PPL: {:.2f}".format(overall_ppl))
print("....................................................")
print("Unemployment PPL: {:.2f}".format(unemployment_ppl))
print("Unemployment AUC: {:.2f}".format(unemployment_auc))
print("....................................................")
print("Out of Labor Force PPL: {:.2f}".format(out_of_labor_force_ppl))
print("Out of Labor Force AUC: {:.2f}".format(out_of_labor_force_auc))
print("....................................................")
print("Education PPL: {:.2f}".format(education_ppl))
print("Education AUC: {:.2f}".format(education_auc))
print("....................................................")
