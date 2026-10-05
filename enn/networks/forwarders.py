# pylint: disable=g-bad-file-header
# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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

"""Modules for forwarding ENNs multiple times."""

import chex
from enn import base
from enn.networks import utils
import haiku as hk
import jax
import typing_extensions


# TODO(author2): clarify the expected shapes/conventions for EnnBatchFwd
class EnnBatchFwd(typing_extensions.Protocol[base.Input]):
  """Creates a sampler for *multiple* logits samples from ENN.

  In most of our code applications this should output something with shape:
    [num_enn_samples, num_batch, num_class]
  However, we are not currently careful/clear about shape expectations, and
  intend to improve on this.
  """

  def __call__(
      self,
      params: hk.Params,
      state: hk.State,
      inputs: base.Input,
  ) -> chex.Array:
    """Forwards the ENN at given inputs for *multiple* index samples."""


def make_batch_fwd(
    enn: base.EpistemicNetwork[base.Input, chex.Array],
    num_enn_samples: int = 1000,
    seed: int = 66,
) -> EnnBatchFwd[base.Input]:
  """Forwards ENN for num_enn_samples sample logits."""

  keys = jax.random.split(jax.random.PRNGKey(seed), num_enn_samples)

  def forward(params: hk.Params, state: hk.State, x: base.Input) -> chex.Array:
    batch_apply = jax.vmap(enn.apply, in_axes=[None, None, None, 0])
    indices = jax.vmap(enn.indexer)(keys)
    net_out, unused_state = batch_apply(params, state, x, indices)
    return utils.parse_net_output(net_out)

  return jax.jit(forward)
