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
"""Network definitions for ResNet."""

import chex
from enn import base
from enn import utils
from enn.networks import ensembles
from enn.networks.resnet import lib
import haiku as hk
import jax
import jax.numpy as jnp


def resnet_model(
    num_output_classes: int,
    enable_double_transpose: bool = True,
    config: lib.ResNetConfig = lib.CanonicalResNets.RESNET_50.value,
) -> lib.ForwardFn:
  """Returns forward network for ResNet."""
  model = lib.ResNet(num_output_classes, config)
  should_transpose_images = (
      enable_double_transpose and jax.local_devices()[0].platform == 'tpu')

  def forward_fn(inputs: base.Array,
                 is_training: bool,
                 test_local_stats: bool = False) -> base.OutputWithPrior:
    # If enabled, there should be a matching NHWC->HWCN transpose in the data.
    if should_transpose_images:
      inputs = jnp.transpose(inputs, (3, 0, 1, 2))  # HWCN -> NHWC

    net_out = model(inputs, is_training, test_local_stats)
    return utils.parse_to_output_with_prior(net_out)

  return forward_fn


class EnsembleResNetENN(base.EpistemicNetworkWithState):
  """Ensemble of ResNet Networks created using einsum ensemble."""

  def __init__(self,
               num_output_classes: int,
               num_ensemble: int = 1,
               is_training: bool = True,
               test_local_stats: bool = False,
               enable_double_transpose: bool = True,
               config: lib.ResNetConfig = lib.CanonicalResNets.RESNET_50.value):
    def net_fn(x: chex.Array) -> base.OutputWithPrior:
      forward_fn = resnet_model(num_output_classes=num_output_classes,
                                enable_double_transpose=enable_double_transpose,
                                config=config)
      net_out = forward_fn(x, is_training, test_local_stats)
      return utils.parse_to_output_with_prior(net_out)
    transformed = hk.without_apply_rng(hk.transform_with_state(net_fn))

    enn = ensembles.EnsembleWithState(transformed, num_ensemble)
    super().__init__(enn.apply, enn.init, enn.indexer)
