# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II

import pdb

@dataclass
class CrossEntropyCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")
    # added for CAREER
    two_stage: bool = field(
        default=False, 
        metadata={"help": "if True, make predictions in two stages: first "
                          "use the transformer representation to predict "
                          "whether someone changes jobs; then marginalize "
                          "over this prediction to predict the job at the "
                          "next timestep."},
    )


@register_criterion("cross_entropy", dataclass=CrossEntropyCriterionConfig)
class CrossEntropyCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg, two_stage):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.two_stage = two_stage

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        # If making predictions in two stages, we must also pass in the 
        # previous tokens.
        prev_tokens = (
            sample['net_input']['src_tokens'] if self.two_stage else None
        )
        loss, _ = self.compute_loss(
            model, net_output, sample, reduce=reduce, prev_tokens=prev_tokens,
        )
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        # Subtract the number of sequences in the batch to account for the fact
        # that we're not including <eos> in the loss.
        batch_size = sample["target"].size(0)
        sample_size -= batch_size
        ntokens = sample["ntokens"] - batch_size
        logging_output = {
            "loss": loss.data,
            "ntokens": ntokens,
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True, 
                     prev_tokens=None,):
        lprobs = model.get_normalized_probs(
            net_output, log_probs=True, two_stage=self.two_stage, 
            prev_tokens=prev_tokens)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        # Replace <eos> with <pad>, since we don't need to predict when a
        # trajectory ends.
        target = target.masked_fill(target == self.eos_idx, self.padding_idx)
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none",
        )
        return loss, loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
