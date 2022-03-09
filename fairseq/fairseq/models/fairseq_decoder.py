# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple

import torch.nn as nn
from fairseq import utils
from torch import Tensor, sigmoid


class FairseqDecoder(nn.Module):
    """Base class for decoders."""

    def __init__(self, dictionary):
        super().__init__()
        self.dictionary = dictionary
        self.onnx_trace = False
        self.adaptive_softmax = None

    def forward(self, prev_output_tokens, encoder_out=None, **kwargs):
        """
        Args:
            prev_output_tokens (LongTensor): shifted output tokens of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (dict, optional): output from the encoder, used for
                encoder-side attention

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens, encoder_out=encoder_out, **kwargs
        )
        x = self.output_layer(x)
        return x, extra

    def extract_features(self, prev_output_tokens, encoder_out=None, **kwargs):
        """
        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        raise NotImplementedError

    def output_layer(self, features, **kwargs):
        """
        Project features to the default output size, e.g., vocabulary size.

        Args:
            features (Tensor): features returned by *extract_features*.
        """
        raise NotImplementedError

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
        two_stage: bool = False,
        prev_tokens: Optional[Tensor] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(
          net_output, log_probs, sample, 
          two_stage=two_stage, prev_tokens=prev_tokens)

    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def get_normalized_probs_scriptable(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
        two_stage: bool = False,
        prev_tokens: Optional[Tensor] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""

        if hasattr(self, "adaptive_softmax") and self.adaptive_softmax is not None:
            if sample is not None:
                assert "target" in sample
                target = sample["target"]
            else:
                target = None
            out = self.adaptive_softmax.get_log_prob(net_output[0], target=target)
            return out.exp_() if not log_probs else out

        logits = net_output[0]
        if two_stage:
            # If we're doing two-stage prediction, the last column of the 
            # logits tensor is reserved for predicting the first stage 
            # transition probabilities (whether or not someone changes jobs).
            stay_logit = logits[:, :, -1]
            stay_prob = sigmoid(stay_logit)
            transition_logits = logits[:, :, :-1]
            # Replace the logits of the previous job with -inf, so that 
            # softmax computes the probabilities conditioned on a transition.
            transition_logits.scatter_(dim=2, index=prev_tokens.unsqueeze(2), 
                                       value=-float("inf"))
            # transition_lprobs contains the log probs conditioned on a transition.
            transition_lprobs = utils.log_softmax(transition_logits, dim=-1)
            # Adjust the transition log probs to account for the probability of
            # staying in the same job.
            transition_lprobs = transition_lprobs + (1. - stay_prob[..., None]).log()
            # Fill in the previous job log probs with the non-transition 
            # probabilities.
            lprobs = transition_lprobs.scatter(
              dim=2, index=prev_tokens.unsqueeze(2), 
              src=stay_prob.log()[..., None].to(transition_lprobs))
            if log_probs:
                return lprobs
            else:
                return lprobs.exp()
        else:
            if log_probs:
                return utils.log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
            else:
                return utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace)

    def max_positions(self):
        """Maximum input length supported by the decoder."""
        return 1e6  # an arbitrary large number

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade old state dicts to work with newer code."""
        return state_dict

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True
