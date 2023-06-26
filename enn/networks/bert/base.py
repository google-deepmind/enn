# pylint: disable=g-bad-file-header
# Copyright 2023 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Base for BERT model."""

import enum
import typing as tp

import chex
from enn import base as enn_base
from enn.networks import base as networks_base
import numpy as np


class BertInput(tp.NamedTuple):
  """Input for the BERT model."""
  token_ids: np.ndarray
  segment_ids: np.ndarray
  input_mask: np.ndarray
  extra: tp.Dict[str, chex.Array] = {}  # You can put other optional stuff here


# Enn modules specialized to work with BertInput.
BertEnn = enn_base.EpistemicNetwork[BertInput, networks_base.OutputWithPrior]
BertApply = enn_base.ApplyFn[BertInput, networks_base.OutputWithPrior]
BertInit = enn_base.InitFn[BertInput]


# Minimal BertConfig copied from
# https://github.com/google-research/bert/blob/master/modeling.py#L31
class BertConfig:
  """Configuration for the BERT Model."""

  def __init__(self,
               vocab_size,
               hidden_size=768,
               num_hidden_layers=12,
               num_attention_heads=12,
               intermediate_size=3072,
               hidden_act='gelu',
               hidden_dropout_prob=0.1,
               attention_probs_dropout_prob=0.1,
               max_position_embeddings=512,
               type_vocab_size=16,
               initializer_range=0.02):
    """Constructs BertConfig.

    Args:
      vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
      hidden_size: Size of the encoder layers and the pooler layer.
      num_hidden_layers: Number of hidden layers in the Transformer encoder.
      num_attention_heads: Number of attention heads for each attention layer in
        the Transformer encoder.
      intermediate_size: The size of the "intermediate" (i.e., feed-forward)
        layer in the Transformer encoder.
      hidden_act: The non-linear activation function (function or string) in the
        encoder and pooler.
      hidden_dropout_prob: The dropout probability for all fully connected
        layers in the embeddings, encoder, and pooler.
      attention_probs_dropout_prob: The dropout ratio for the attention
        probabilities.
      max_position_embeddings: The maximum sequence length that this model might
        ever be used with. Typically set this to something large just in case
        (e.g., 512 or 1024 or 2048).
      type_vocab_size: The vocabulary size of the `token_type_ids` passed into
        `BertModel`.
      initializer_range: The stdev of the truncated_normal_initializer for
        initializing all weight matrices.
    """
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.hidden_act = hidden_act
    self.intermediate_size = intermediate_size
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.max_position_embeddings = max_position_embeddings
    self.type_vocab_size = type_vocab_size
    self.initializer_range = initializer_range


def bert_small() -> BertConfig:
  """Config for small BERT with ~110M params."""
  return BertConfig(
      attention_probs_dropout_prob=0.1,
      hidden_act='gelu',
      hidden_dropout_prob=0.1,
      hidden_size=768,
      initializer_range=0.02,
      intermediate_size=3072,
      max_position_embeddings=512,
      num_attention_heads=12,
      num_hidden_layers=12,
      type_vocab_size=2,
      vocab_size=30522,
  )


def bert_large() -> BertConfig:
  """Config for large BERT with ~340M params."""
  return BertConfig(
      attention_probs_dropout_prob=0.1,
      hidden_act='gelu',
      hidden_dropout_prob=0.1,
      hidden_size=1024,
      initializer_range=0.02,
      intermediate_size=4096,
      max_position_embeddings=512,
      num_attention_heads=16,
      num_hidden_layers=24,
      type_vocab_size=2,
      vocab_size=30522)


class BertConfigs(enum.Enum):
  """Configs for BERT models."""
  BERT_SMALL: BertConfig = bert_small()  # ~110M params
  BERT_LARGE: BertConfig = bert_large()  # ~340M params
