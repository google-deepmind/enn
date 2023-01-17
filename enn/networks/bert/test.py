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

"""Tests for BERT ENN."""
from absl.testing import absltest
from absl.testing import parameterized
from enn import base as enn_base
from enn.networks import utils as networks_utils
from enn.networks.bert import base
from enn.networks.bert import bert
import haiku as hk
import jax
import jax.numpy as jnp


def _fake_data(seed: int,
               num_train: int,
               num_classes: int,
               sequence_len: int) -> enn_base.Batch:
  """Generates a fake dataset."""
  rng = hk.PRNGSequence(seed)

  token_ids = jax.random.randint(
      next(rng), [num_train, sequence_len], 0, 10_000)
  segment_ids = jnp.zeros([num_train, sequence_len], jnp.int32)
  input_mask = jnp.ones([num_train, sequence_len], jnp.int32)

  batch_start = jax.random.randint(next(rng), [], 0, 1_000_000)
  data_index = jnp.arange(num_train) + batch_start

  return enn_base.Batch(
      x=base.BertInput(token_ids, segment_ids, input_mask),
      y=jax.random.randint(next(rng), [num_train], 0, num_classes),
      data_index=data_index,
  )


class NetworkTest(parameterized.TestCase):
  """Tests for the BERT model."""

  @parameterized.product(
      output_size=[3, 6],
      num_train=[1, 10],
      is_training=[True, False],
  )
  def test_forward_pass(
      self,
      output_size: int,
      num_train: int,
      is_training: bool,
  ):
    """Tests forward pass and output shape."""
    bert_config = base.BertConfig(
        vocab_size=128,
        num_hidden_layers=2,
        num_attention_heads=3,
        hidden_size=output_size,
    )
    bert_enn = bert.make_bert_enn(
        bert_config=bert_config, is_training=is_training
    )
    fake_batch = _fake_data(
        seed=0,
        num_train=num_train,
        num_classes=output_size,
        sequence_len=128,
    )
    rng = hk.PRNGSequence(0)
    index = bert_enn.indexer(next(rng))
    params, state = bert_enn.init(next(rng), fake_batch.x, index)
    out, unused_new_state = bert_enn.apply(params, state, fake_batch.x, index)
    logits = networks_utils.parse_net_output(out)
    self.assertEqual(logits.shape, (num_train, output_size))


if __name__ == '__main__':
  absltest.main()
