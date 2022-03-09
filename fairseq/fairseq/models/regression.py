"""A 'regression' model for labor data."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from fairseq.dataclass import FairseqDataclass
from fairseq.models import (
    FairseqLanguageModel,
    FairseqIncrementalDecoder,
    register_model,
)
from fairseq.models.transformer import Embedding
from torch import Tensor
import torch

import pdb


@dataclass
class RegressionModelConfig(FairseqDataclass):
  second_order_markov: bool = field(
    default=False, 
    metadata={"help": "whether to include the second-to-last job"}
  )
  include_years_in_current_job: bool = field(
    default=False, 
    metadata={"help": "whether to include the number of years in the current "
                      "job"}
  )
  include_total_years: bool = field(
    default=False, 
    metadata={"help": "whether to include the total number of years"}
  )
  include_year: bool = field(
    default=False, 
    metadata={"help": "whether to include a fixed effect for the most recent "
                      "year"}
  )
  non_consecutive_year_effect: bool = field(
    default=False, 
    metadata={"help": "whether to include a dummy indicating that a year is "
                      "non-consecutive from the previous year"}
  )
  include_education: bool = field(
    default=False, 
    metadata={"help": "whether to include the most recent education"}
  )
  education_difference: bool = field(
    default=False, 
    metadata={"help": "whether to include an indicator for whether someone's "
                      "education changes"}
  )
  include_ethnicity: bool = field(
    default=False, 
    metadata={"help": "whether to include race/ethnicity"}
  )
  include_gender: bool = field(
    default=False,
    metadata={"help": "whether to include gender"}
  )
  include_location: bool = field(
    default=False, 
    metadata={"help": "whether to include location"}
  )
  two_stage: Optional[bool] = field(
    default=False, 
    metadata={"help": "if True, use two-stage training"}
  )

@register_model("regression", dataclass=RegressionModelConfig)
class RegressionModel(FairseqLanguageModel):
  def __init__(self, decoder):
    super().__init__(decoder)
  
  @classmethod
  def build_model(cls, args, task):
    """Build a new model instance."""
    embed_dim = (len(task.target_dictionary) + 1 if args.two_stage 
                 else len(task.target_dictionary))
    embed_bias = torch.nn.Parameter(torch.zeros(embed_dim))
    embed_current_job = cls.build_embedding(
      args, task.target_dictionary, embed_dim)
    embed_previous_job = cls.build_embedding(
      args, task.target_dictionary, embed_dim)
    embed_years_in_current_job = torch.nn.Parameter(
      torch.zeros(embed_dim)) if args.include_years_in_current_job else None
    embed_total_years = torch.nn.Parameter(
      torch.zeros(embed_dim)) if args.include_total_years else None
    embed_year = cls.build_embedding(
      args, task._year_dictionary, embed_dim) if args.include_year else None
    embed_non_consecutive_year_effect = torch.nn.Parameter(
      torch.zeros(embed_dim)) if args.non_consecutive_year_effect else None
    embed_education = (cls.build_embedding(
      args, task._education_dictionary, embed_dim) if args.include_education 
      else None)
    embed_education_difference = torch.nn.Parameter(
      torch.zeros(embed_dim)) if args.education_difference else None
    embed_ethnicity = (cls.build_embedding(
      args, task._ethnicity_dictionary, embed_dim) if args.include_ethnicity 
      else None)
    embed_gender = (cls.build_embedding(
      args, task._gender_dictionary, embed_dim) if args.include_gender 
      else None)
    embed_location = (cls.build_embedding(
      args, task._location_dictionary, embed_dim) if args.include_location 
      else None)

    decoder = RegressionDecoder(
      args, task.target_dictionary, task._year_dictionary, embed_bias,
      embed_current_job, embed_previous_job, embed_years_in_current_job, 
      embed_total_years, embed_year, embed_non_consecutive_year_effect, 
      embed_education, embed_education_difference, embed_ethnicity, 
      embed_gender, embed_location)
    return cls(decoder)
  
  @classmethod
  def build_embedding(cls, args, dictionary, embed_dim, path=None):
    embed_tokens = Embedding(len(dictionary), embed_dim, dictionary.pad())
    return embed_tokens


class RegressionDecoder(FairseqIncrementalDecoder):
  def __init__(self, args, dictionary, year_dictionary, embed_bias, 
               embed_current_job, embed_previous_job, 
               embed_years_in_current_job, embed_total_years, embed_year, 
               embed_non_consecutive_year_effect, embed_education, 
               embed_education_difference, embed_ethnicity, embed_gender, 
               embed_location):
    self.args = args
    super().__init__(dictionary)
    self.year_dictionary = year_dictionary
    
    self.embed_bias = embed_bias
    self.embed_current_job = embed_current_job
    self.embed_previous_job = embed_previous_job
    self.embed_years_in_current_job = embed_years_in_current_job
    self.embed_total_years = embed_total_years
    self.embed_year = embed_year
    self.embed_non_consecutive_year_effect = embed_non_consecutive_year_effect
    self.embed_education = embed_education
    self.embed_education_difference = embed_education_difference
    self.embed_ethnicity = embed_ethnicity
    self.embed_gender = embed_gender
    self.embed_location = embed_location
    self.two_stage = args.two_stage

    self.padding_idx = embed_current_job.padding_idx

    self.second_order_markov = self.args.second_order_markov
    self.include_years_in_current_job = self.args.include_years_in_current_job
    self.include_total_years = self.args.include_total_years
    self.include_year = self.args.include_year
    self.non_consecutive_year_effect = self.args.non_consecutive_year_effect
    self.include_education = self.args.include_education
    self.education_difference = self.args.education_difference
    self.include_ethnicity = self.args.include_ethnicity
    self.include_gender = self.args.include_gender
    self.include_location = self.args.include_location

  def forward(self, 
              prev_output_tokens,
              encoder_out: Optional[Dict[str, List[Tensor]]] = None,
              incremental_state: Optional[Dict[str, Dict[str, Tensor]]] = None,
              years: Optional[torch.tensor] = None,
              educations: Optional[torch.tensor] = None,
              ethnicities: Optional[torch.tensor] = None,
              genders: Optional[torch.tensor] = None,
              locations: Optional[torch.tensor] = None,
              **unused):
    """
    Args:
      prev_output_tokens: previous decoder outputs of shape [batch, tgt_len]
    
    Returns:
      tuple:
        - the decoder's output of shape [batch, tgt_len, vocab]
        - a dictionary with any model-specific outputs
    """
    logits = self.embed_current_job(prev_output_tokens)
    bias = self.embed_bias[None, None, :]
    logits += bias
    if self.second_order_markov:
      two_jobs_ago = torch.nn.functional.pad(
        prev_output_tokens, (1, 0, 0, 0), 
        value=self.padding_idx)[:, :-1]
      logits += self.embed_previous_job(two_jobs_ago)
    if self.include_years_in_current_job:
      years_in_current_job = torch.zeros_like(prev_output_tokens)
      seq_len = prev_output_tokens.size(1)
      for idx in range(1, seq_len):
        years_in_current_job[:, idx] = torch.where(
          prev_output_tokens[:, idx] == prev_output_tokens[:, idx - 1], 
          years_in_current_job[:, idx - 1] + 1, 
          0)
      years_in_current_job_effects = years_in_current_job.float()[
        :, :, None] * self.embed_years_in_current_job[None, None, :]
      logits += years_in_current_job_effects
    if self.include_total_years:
      total_year_effects = torch.arange(
        seq_len)[None, :, None].to(logits) * self.embed_total_years[
          None, None, :]
      logits += total_year_effects
    if self.include_year:
      year_effects = self.embed_year(years)
      logits += year_effects
    if self.non_consecutive_year_effect:
      # NOTE: This code assumes that consecutive years have consecutive 
      # indices in the year dictionary.
      year_diffs = years[:, 1:] - years[:, :-1]
      year_diffs = torch.nn.functional.pad(year_diffs, (1, 0, 0, 0), value=1.)
      year_diffs = (year_diffs != 1.).float()
      non_consecutive_year_effect = (
        year_diffs.float()[:, :, None] * 
        self.embed_non_consecutive_year_effect[None, None, :])
      logits += non_consecutive_year_effect
    if self.include_education:
      education_effects = self.embed_education(educations)
      logits += education_effects
    if self.education_difference:
      new_education = (educations[:, 1:] != educations[:, :-1]).float()
      new_education = torch.nn.functional.pad(
        new_education, (1, 0, 0, 0), value=1.)
      education_difference_effects = new_education.float()[
        :, :, None] * self.embed_education_difference[None, None, :]
      logits += education_difference_effects
    if self.include_ethnicity:
      ethnicity_effects = self.embed_ethnicity(ethnicities)
      logits += ethnicity_effects
    if self.include_gender:
      gender_effects = self.embed_gender(genders)
      logits += gender_effects
    if self.include_location:
      location_effects = self.embed_location(locations)
      logits += location_effects
    # Clamp for numerical stability.
    logits = torch.clamp(logits, -10, 10)
    return logits, {}
