# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from fairseq import utils
from fairseq.data import (
    AppendTokenDataset,
    Dictionary,
    IdDataset,
    LMContextWindowDataset,
    MonolingualDataset,
    NestedDictionaryDataset,
    NumelDataset,
    OffsetTokensDataset,
    PadDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TokenBlockDataset,
    TruncatedDictionary,
    WorkerDataset,
    data_utils,
)
from fairseq.data.indexed_dataset import get_available_dataset_impl
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.tasks import LegacyFairseqTask, register_task
from omegaconf import II

import pdb

SAMPLE_BREAK_MODE_CHOICES = ChoiceEnum(["none", "complete", "complete_doc", "eos"])
SHORTEN_METHOD_CHOICES = ChoiceEnum(["none", "truncate", "random_crop"])
logger = logging.getLogger(__name__)


@dataclass
class OccupationModelingConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None, metadata={"help": "path to data directory"}
    )
    sample_break_mode: SAMPLE_BREAK_MODE_CHOICES = field(
        default="none",
        metadata={
            "help": 'If omitted or "none", fills each sample with tokens-per-sample '
            'tokens. If set to "complete", splits samples only at the end '
            "of sentence, but may include multiple sentences per sample. "
            '"complete_doc" is similar but respects doc boundaries. '
            'If set to "eos", includes only one sentence per sample.'
        },
    )
    tokens_per_sample: int = field(
        default=1024,
        metadata={"help": "max number of tokens per sample for LM dataset"},
    )
    output_dictionary_size: int = field(
        default=-1, metadata={"help": "limit the size of output dictionary"}
    )
    self_target: bool = field(default=False, metadata={"help": "include self target"})
    future_target: bool = field(
        default=False, metadata={"help": "include future target"}
    )
    past_target: bool = field(default=False, metadata={"help": "include past target"})
    add_bos_token: bool = field(
        default=False, metadata={"help": "prepend beginning of sentence token (<s>)"}
    )
    max_target_positions: Optional[int] = field(
        default=None, metadata={"help": "max number of tokens in the target sequence"}
    )
    shorten_method: SHORTEN_METHOD_CHOICES = field(
        default="none",
        metadata={
            "help": "if not none, shorten sequences that exceed --tokens-per-sample"
        },
    )
    shorten_data_split_list: str = field(
        default="",
        metadata={
            "help": "comma-separated list of dataset splits to apply shortening to, "
            'e.g., "train,valid" (default: all dataset splits)'
        },
    )
    pad_to_fixed_length: Optional[bool] = field(
        default=False,
        metadata={"help": "pad to fixed length"},
    )
    pad_to_fixed_bsz: Optional[bool] = field(
        default=False,
        metadata={"help": "boolean to pad to fixed batch size"},
    )
    ## Added for CAREER
    include_year: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to include year covariate"},
    )
    include_education: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to include education covariate"},
    )
    include_ethnicity: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to include race/ethnicity covariate"},
    )
    include_gender: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to include gender covariate"},
    )
    include_location: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to include location covariate"},
    )

    # TODO common vars below add to parent
    seed: int = II("common.seed")
    batch_size: Optional[int] = II("dataset.batch_size")
    batch_size_valid: Optional[int] = II("dataset.batch_size_valid")
    dataset_impl: Optional[ChoiceEnum(get_available_dataset_impl())] = II(
        "dataset.dataset_impl"
    )
    data_buffer_size: int = II("dataset.data_buffer_size")
    tpu: bool = II("common.tpu")
    use_plasma_view: bool = II("common.use_plasma_view")
    plasma_path: str = II("common.plasma_path")


@register_task("occupation_modeling", dataclass=OccupationModelingConfig)
class OccupationModelingTask(LegacyFairseqTask):
    """
    Train an occupation model.

    Args:
        dictionary (~fairseq.data.Dictionary): the dictionary for the input of
            the occupation model
        output_dictionary (~fairseq.data.Dictionary): the dictionary for the
            output of the occupation model. In most cases it will be the same as
            *dictionary*, but could possibly be a more limited version of the
            dictionary (if ``--output-dictionary-size`` is used).
        targets (List[str]): list of the target types that the occupation model
            should predict.  Can be one of "self", "future", and "past".
            Defaults to "future".

    .. note::

        The occupation modeling task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate`, :mod:`fairseq-interactive` and
        :mod:`fairseq-eval-lm`.

    The occupation modeling task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.occupation_modeling_parser
        :prog:
    """

    def __init__(self, args, dictionary, year_dictionary, education_dictionary,
                 ethnicity_dictionary, gender_dictionary, location_dictionary, 
                 output_dictionary=None, targets=None):
        super().__init__(args)
        self.dictionary = dictionary
        self._job_dictionary = dictionary
        self._year_dictionary = year_dictionary
        self._education_dictionary = education_dictionary
        self._ethnicity_dictionary = ethnicity_dictionary
        self._gender_dictionary = gender_dictionary
        self._location_dictionary = location_dictionary
        self.output_dictionary = output_dictionary or dictionary

        if targets is None:
            targets = ["future"]
        self.targets = targets

    @classmethod
    def setup_dictionary(cls, args, **kwargs):
        def load_dictionary(covariate):
            try:
                return Dictionary.load(
                  os.path.join(paths[0], "{}/dict.txt".format(covariate)))
            except:
                print("No dictionary found for {}, so loading jobs dictionary instead".format(
                  covariate))
                return Dictionary.load(
                  os.path.join(paths[0], "job/dict.txt".format(covariate)))

        dictionary = None
        output_dictionary = None
        if args.data:
            paths = utils.split_paths(args.data)
            assert len(paths) > 0
            dictionary = Dictionary.load(os.path.join(paths[0], "job/dict.txt"))
            year_dictionary = load_dictionary("year")
            education_dictionary = load_dictionary("education")
            gender_dictionary = load_dictionary("gender")
            ethnicity_dictionary = load_dictionary("ethnicity")
            location_dictionary = load_dictionary("location")
            logger.info("dictionary: {} types".format(len(dictionary)))
            output_dictionary = dictionary
            if args.output_dictionary_size >= 0:
                output_dictionary = TruncatedDictionary(
                    dictionary, args.output_dictionary_size
                )
        return (dictionary, year_dictionary, education_dictionary, 
                ethnicity_dictionary, gender_dictionary, location_dictionary, 
                output_dictionary)

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        (dictionary, year_dictionary, education_dictionary, 
         ethnicity_dictionary, gender_dictionary, location_dictionary, 
         output_dictionary) = cls.setup_dictionary(args, **kwargs)

        # upgrade old checkpoints
        if getattr(args, "exclude_self_target", False):
            args.self_target = False

        targets = []
        if getattr(args, "self_target", False):
            targets.append("self")
        if getattr(args, "future_target", False):
            targets.append("future")
        if getattr(args, "past_target", False):
            targets.append("past")
        if len(targets) == 0:
            # standard occupation modeling
            targets = ["future"]

        return cls(args, dictionary, year_dictionary, education_dictionary, 
                   ethnicity_dictionary, gender_dictionary, location_dictionary, 
                   output_dictionary, targets=targets)

    def build_model(self, args, from_checkpoint=False):
        model = super().build_model(args, from_checkpoint)
        for target in self.targets:
            if target not in model.supported_targets:
                raise ValueError(
                    "Unsupported occupation modeling target: {}".format(target)
                )

        return model

    def load_dataset(
        self, split: str, epoch=1, combine=False, **kwargs
    ) -> WorkerDataset:
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, valid1, test)
        """
        def load_time_varying_dataset(name):
            if name == 'job' or getattr(self.args, "include_{}".format(name)):
                paths = utils.split_paths(os.path.join(self.args.data, name))
                assert len(paths) > 0

                data_path = paths[(epoch - 1) % len(paths)]
                split_path = os.path.join(data_path, split)

                # each process has its own copy of the raw data (likely to be an np.memmap)
                dataset = data_utils.load_indexed_dataset(
                    split_path, getattr(self, "_" + name + "_dictionary"), 
                    self.args.dataset_impl, combine=combine,
                )
                if dataset is None:
                    if name == 'job':
                        raise FileNotFoundError(
                            "No job dataset found at {}/{}".format(split, data_path))
                    elif getattr(self.args, "include_{}".format(name)):
                        raise FileNotFoundError(
                            "Using '--include-{}' flag but no binary data found in "
                            "{}. You can either remove the '--include-{}' flag to "
                            "not use this covariate or preprocess {} data in {}".format(
                              name, data_path, name, name, data_path))

                dataset = maybe_shorten_dataset(
                    dataset,
                    split,
                    self.args.shorten_data_split_list,
                    self.args.shorten_method,
                    self.args.tokens_per_sample,
                    self.args.seed,
                )
                dataset = TokenBlockDataset(
                    dataset,
                    dataset.sizes,
                    self.args.tokens_per_sample,
                    pad=getattr(self, "_" + name + "_dictionary").pad(),
                    eos=getattr(self, "_" + name + "_dictionary").eos(),
                    break_mode=self.args.sample_break_mode,
                    include_targets=True,
                    use_plasma_view=self.args.use_plasma_view,
                    split_path=split_path,
                    plasma_path=self.args.plasma_path,
                )
                return dataset
            else:
                return None
        
        def load_static_dataset(name):
            if getattr(self.args, "include_{}".format(name)):
                paths = utils.split_paths(os.path.join(self.args.data, name))
                data_path = paths[(epoch - 1) % len(paths)]
                split_path = os.path.join(data_path, split)
                dataset = data_utils.load_indexed_dataset(
                    split_path, getattr(self, "_" + name + "_dictionary"), 
                    self.args.dataset_impl, combine=combine,
                )
                if dataset is None:
                    raise FileNotFoundError(
                        "Using '--include-{}' flag but no binary data found in "
                        "{}{}. You can either remove the '--include-{}' flag to "
                        "not use this covariate or preprocess {} data in {}".format(
                          name, split, data_path, name, name, data_path))

                dataset = maybe_shorten_dataset(
                    dataset,
                    split,
                    self.args.shorten_data_split_list,
                    self.args.shorten_method,
                    self.args.tokens_per_sample,
                    self.args.seed,
                )
                dataset = OffsetTokensDataset(
                  StripTokenDataset(
                      dataset,
                      id_to_strip=getattr(self, "_" + name + "_dictionary").eos(),
                  ),
                    offset=0,
                )
                return dataset
            else:
                return None

        job_dataset = load_time_varying_dataset("job")
        year_dataset = load_time_varying_dataset("year")
        education_dataset = load_time_varying_dataset("education")
        ethnicity_dataset = load_static_dataset("ethnicity")
        gender_dataset = load_static_dataset("gender")
        location_dataset = load_static_dataset("location")

        add_eos_for_other_targets = (
            self.args.sample_break_mode is not None
            and self.args.sample_break_mode != "none"
        )
        fixed_pad_length = None
        if self.args.pad_to_fixed_length:
            fixed_pad_length = self.args.tokens_per_sample

        pad_to_bsz = None
        if self.args.pad_to_fixed_bsz:
            pad_to_bsz = (
                self.args.batch_size_valid if "valid" in split else self.args.batch_size
            )

        self.datasets[split] = WorkerDataset(
            job_dataset=job_dataset,
            year_dataset=year_dataset,
            education_dataset=education_dataset,
            ethnicity_dataset=ethnicity_dataset,
            gender_dataset=gender_dataset,
            location_dataset=location_dataset,
            sizes=job_dataset.sizes,
            src_vocab=self.dictionary,
            tgt_vocab=self.output_dictionary,
            add_eos_for_other_targets=add_eos_for_other_targets,
            shuffle=True,
            targets=self.targets,
            add_bos_token=self.args.add_bos_token,
            fixed_pad_length=fixed_pad_length,
            pad_to_bsz=pad_to_bsz,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        """
        Generate batches for inference. We prepend an eos token to src_tokens
        (or bos if `--add-bos-token` is set) and we append a <pad> to target.
        This is convenient both for generation with a prefix and LM scoring.
        """
        dataset = StripTokenDataset(
            TokenBlockDataset(
                src_tokens,
                src_lengths,
                block_size=None,  # ignored for "eos" break mode
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode="eos",
            ),
            # remove eos from (end of) target sequence
            self.source_dictionary.eos(),
        )
        src_dataset = PrependTokenDataset(
            dataset,
            token=(
                self.source_dictionary.bos()
                if getattr(self.args, "add_bos_token", False)
                else self.source_dictionary.eos()
            ),
        )
        tgt_dataset = AppendTokenDataset(dataset, token=self.source_dictionary.pad())
        return NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": {
                    "src_tokens": PadDataset(
                        src_dataset,
                        pad_idx=self.source_dictionary.pad(),
                        left_pad=False,
                    ),
                    "src_lengths": NumelDataset(src_dataset, reduce=False),
                },
                "target": PadDataset(
                    tgt_dataset, pad_idx=self.source_dictionary.pad(), left_pad=False
                ),
            },
            sizes=[np.array(src_lengths)],
        )

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        with torch.no_grad():
            # Generation will always be conditioned on bos_token
            if getattr(self.args, "add_bos_token", False):
                bos_token = self.source_dictionary.bos()
            else:
                bos_token = self.source_dictionary.eos()

            if constraints is not None:
                raise NotImplementedError(
                    "Constrained decoding with the occupation_modeling task is not supported"
                )

            # SequenceGenerator doesn't use src_tokens directly, we need to
            # pass the `prefix_tokens` argument instead
            if prefix_tokens is None and sample["net_input"]["src_tokens"].nelement():
                prefix_tokens = sample["net_input"]["src_tokens"]
                if prefix_tokens[:, 0].eq(bos_token).all():
                    prefix_tokens = prefix_tokens[:, 1:]

            return generator.generate(
                models, sample, prefix_tokens=prefix_tokens, bos_token=bos_token
            )

    def eval_lm_dataloader(
        self,
        dataset,
        max_tokens: Optional[int] = 36000,
        batch_size: Optional[int] = None,
        max_positions: Optional[int] = None,
        num_shards: int = 1,
        shard_id: int = 0,
        num_workers: int = 1,
        data_buffer_size: int = 10,
        # ensures that every evaluated token has access to a context of at least
        # this size, if possible
        context_window: int = 0,
    ):
        if context_window > 0:
            dataset = LMContextWindowDataset(
                dataset=dataset,
                tokens_per_sample=self.args.tokens_per_sample,
                context_window=context_window,
                pad_idx=self.source_dictionary.pad(),
            )
        return self.get_batch_iterator(
            dataset=dataset,
            max_tokens=max_tokens,
            max_sentences=batch_size,
            max_positions=max_positions,
            ignore_invalid_inputs=True,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            data_buffer_size=data_buffer_size,
        ).next_epoch_itr(shuffle=False)

    @property
    def source_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the occupation
        model."""
        return self.dictionary

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the occupation
        model."""
        return self.output_dictionary
