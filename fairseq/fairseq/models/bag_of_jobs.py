"""A 'bag of jobs' model for occupation modeling."""
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
class BagOfJobsModelConfig(FairseqDataclass):
  embed_dim: int = field(
    default=1024, metadata={"help": "representaiton dimension"})
  two_stage: Optional[bool] = field(
    default=False, metadata={"help": "if True, use two-stage training"}
  )
  include_year: Optional[bool] = field(
    default=False, metadata={"help": "if True, include year covariates"}
  )
  include_education: Optional[bool] = field(
    default=False, metadata={"help": "if True, include education covariates"}
  )
  include_ethnicity: Optional[bool] = field(
    default=False, metadata={"help": "if True, include ethnicity covariate"}
  )
  include_gender: Optional[bool] = field(
    default=False, metadata={"help": "if True, include gender covariate"}
  )
  include_location: Optional[bool] = field(
    default=False, metadata={"help": "if True, include location covariate"}
  )


@register_model("bag_of_jobs", dataclass=BagOfJobsModelConfig)
class BagOfJobsModel(FairseqLanguageModel):
  def __init__(self, decoder):
    super().__init__(decoder)
  
  @classmethod
  def build_model(cls, args, task):
    """Build a new model instance."""
    embed_current_job = cls.build_embedding(
      args, task.target_dictionary, args.embed_dim)
    embed_context = cls.build_embedding(
      args, task.target_dictionary, args.embed_dim)
    embed_current_year = cls.build_embedding(
      args, task._year_dictionary, args.embed_dim
    ) if args.include_year else None
    embed_context_year = cls.build_embedding(
      args, task._year_dictionary, args.embed_dim
    ) if args.include_year else None
    embed_current_education = cls.build_embedding(
      args, task._education_dictionary, args.embed_dim
    ) if args.include_education else None
    embed_context_education = cls.build_embedding(
      args, task._education_dictionary, args.embed_dim
    ) if args.include_education else None
    embed_ethnicity = cls.build_embedding(
      args, task._ethnicity_dictionary, args.embed_dim
    ) if args.include_ethnicity else None
    embed_gender = cls.build_embedding(
      args, task._gender_dictionary, args.embed_dim
    ) if args.include_gender else None
    embed_location = cls.build_embedding(
      args, task._location_dictionary, args.embed_dim
    ) if args.include_location else None
    decoder = BagOfJobsDecoder(
      args, task.target_dictionary, embed_current_job, embed_context,
      embed_current_year, embed_context_year,
      embed_current_education, embed_context_education,
      embed_ethnicity, embed_gender, embed_location)
    return cls(decoder)
  
  @classmethod
  def build_embedding(cls, args, dictionary, embed_dim, path=None):
    embed_tokens = Embedding(len(dictionary), embed_dim, dictionary.pad())
    return embed_tokens


class BagOfJobsDecoder(FairseqIncrementalDecoder):
  def __init__(self, args, dictionary, embed_current_job, embed_context,
               embed_current_year, embed_context_year,
               embed_current_education, embed_context_education,
               embed_ethnicity, embed_gender, embed_location):
    self.args = args
    super().__init__(dictionary)
    self.embed_dim = args.embed_dim
    self.embed_dim = args.embed_dim
    self.include_year = args.include_year
    self.include_education = args.include_education
    self.include_ethnicity = args.include_ethnicity
    self.include_gender = args.include_gender
    self.include_location = args.include_location
    
    self.embed_current_job = embed_current_job
    self.embed_context = embed_context
    self.embed_current_year = embed_current_year
    self.embed_context_year = embed_context_year
    self.embed_current_education = embed_current_education
    self.embed_context_education = embed_context_education
    self.embed_ethnicity = embed_ethnicity
    self.embed_gender = embed_gender
    self.embed_location = embed_location

    self.padding_idx = embed_context.padding_idx

    self.two_stage = args.two_stage

    output_classes = len(dictionary) + 1 if self.two_stage else len(dictionary)
    self.current_job_output_projection = torch.nn.Linear(
      self.embed_dim, output_classes, bias=False
    )
    self.context_output_projection = torch.nn.Linear(
      self.embed_dim, output_classes, bias=False
    )
    self.current_year_output_projection = torch.nn.Linear(
      self.embed_dim, output_classes, bias=False
    )
    self.context_year_output_projection = torch.nn.Linear(
      self.embed_dim, output_classes, bias=False
    )
    self.current_education_output_projection = torch.nn.Linear(
      self.embed_dim, output_classes, bias=False
    )
    self.context_education_output_projection = torch.nn.Linear(
      self.embed_dim, output_classes, bias=False
    )
    self.ethnicity_output_projection = torch.nn.Linear(
      self.embed_dim, output_classes, bias=False
    )
    self.gender_output_projection = torch.nn.Linear(
      self.embed_dim, output_classes, bias=False
    )
    self.location_output_projection = torch.nn.Linear(
      self.embed_dim, output_classes, bias=False
    )
  
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
    current_job_embeds = self.embed_current_job(prev_output_tokens)
    padded_context = torch.nn.functional.pad(
      prev_output_tokens, (1, 0, 0, 0))[:, :-1]
    # context_embeds.shape == [batch, tgt_len, embed_dim]
    context_embeds = self.embed_context(padded_context)
    context_sum = torch.cumsum(context_embeds, dim=1)
    seq_len = prev_output_tokens.shape[1]
    context_lens = (torch.arange(seq_len) + 1).to(context_sum)
    context_mean = context_sum / context_lens[None, :, None]

    # current_job_logits.shape == [batch, tgt_len, output_classes]
    current_job_logits = self.current_job_output_projection(current_job_embeds)
    context_logits = self.context_output_projection(context_mean)
    
    logits = current_job_logits + context_logits

    if self.include_year:
      current_year_embeds = self.embed_current_year(years)
      padded_context_years = torch.nn.functional.pad(
        years, (1, 0, 0, 0))[:, :-1]
      context_year_embeds = self.embed_context_year(padded_context_years)
      context_year_sum = torch.cumsum(context_year_embeds, dim=1)
      context_year_mean = context_year_sum / context_lens[None, :, None]
      current_year_logits = self.current_year_output_projection(
        current_year_embeds)
      context_year_logits = self.context_year_output_projection(
        context_year_mean)
      logits = logits + current_year_logits + context_year_logits
    
    # pdb.set_trace()
    if self.include_education:
      current_education_embeds = self.embed_current_education(educations)
      padded_context_educations = torch.nn.functional.pad(
        educations, (1, 0, 0, 0))[:, :-1]
      context_education_embeds = self.embed_context_education(
        padded_context_educations)
      context_education_sum = torch.cumsum(context_education_embeds, dim=1)
      context_education_mean = context_education_sum / context_lens[
        None, :, None]
      current_education_logits = self.current_education_output_projection(
        current_education_embeds)
      context_education_logits = self.context_education_output_projection(
        context_education_mean)
      logits = logits + current_education_logits + context_education_logits
    
    if self.include_ethnicity:
      ethnicity_embeds = self.embed_ethnicity(ethnicities)
      ethnicity_logits = self.ethnicity_output_projection(ethnicity_embeds)
      logits += ethnicity_logits
    
    if self.include_gender:
      gender_embeds = self.embed_gender(genders)
      gender_logits = self.gender_output_projection(gender_embeds)
      logits += gender_logits
    
    if self.include_location:
      location_embeds = self.embed_location(locations)
      location_logits = self.location_output_projection(location_embeds)
      logits += location_logits

    # Clamp for numerical stability.
    logits = torch.clamp(logits, min=-10, max=10)
    return logits, {}
