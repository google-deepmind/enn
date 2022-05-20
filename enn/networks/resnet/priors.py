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
"""ResNet with priors for ImageNet."""

from typing import Sequence

import chex
from enn import base
from enn.networks import indexers
from enn.networks.resnet import base as resnet_base
import haiku as hk
import jax
import jax.numpy as jnp


class ResnetMlpPrior(base.EpistemicNetworkWithState):
  """Resnet Network with MLP Prior."""

  def __init__(self,
               num_classes: int,
               prior_scale: float = 1.,
               hidden_sizes: Sequence[int] = (10,),
               is_training: bool = True):

    def net_fn(x: chex. Array, index: base.Index) -> base.OutputWithPrior:
      del index
      output = resnet_base.resnet_model(num_classes)(x, is_training=is_training)

      # MLP Prior
      if jax.local_devices()[0].platform == 'tpu':
        x = jnp.transpose(x, (3, 0, 1, 2))  # HWCN -> NHWC
      x = hk.Flatten()(x)
      prior = hk.nets.MLP(list(hidden_sizes) + [num_classes,], name='prior')(x)
      return base.OutputWithPrior(train=output.train, prior=prior_scale*prior,
                                  extra=output.extra)

    transformed = hk.without_apply_rng(hk.transform_with_state(net_fn))
    enn = base.EpistemicNetworkWithState(
        apply=transformed.apply,
        init=transformed.init,
        indexer=indexers.EnsembleIndexer(1))
    super().__init__(enn.apply, enn.init, enn.indexer)


class ResnetCnnPrior(base.EpistemicNetworkWithState):
  """VGG Network with ConvNet Prior."""

  def __init__(self,
               num_classes: int,
               prior_scale: float = 1.,
               output_channels: Sequence[int] = (4, 8, 8),
               kernel_sizes: Sequence[int] = (10, 10, 3),
               strides: Sequence[int] = (5, 5, 2),
               is_training: bool = True):

    assert len(output_channels) == len(kernel_sizes) == len(strides)

    def net_fn(x: chex. Array, index: base.Index) -> base.OutputWithPrior:
      del index
      output = resnet_base.resnet_model(num_classes)(x, is_training=is_training)

      # CNN Prior
      if jax.local_devices()[0].platform == 'tpu':
        x = jnp.transpose(x, (3, 0, 1, 2))  # HWCN -> NHWC
      for channels, kernel_size, stride in zip(output_channels,
                                               kernel_sizes,
                                               strides,):
        x = hk.Conv2D(
            output_channels=channels,
            kernel_shape=[kernel_size, kernel_size],
            stride=stride,
            name='prior_conv')(x)
        x = jax.nn.relu(x)
      x = hk.Flatten()(x)
      prior = hk.nets.MLP([num_classes], name='prior')(x)
      return base.OutputWithPrior(train=output.train, prior=prior_scale*prior,
                                  extra=output.extra)

    transformed = hk.without_apply_rng(hk.transform_with_state(net_fn))
    enn = base.EpistemicNetworkWithState(
        apply=transformed.apply,
        init=transformed.init,
        indexer=indexers.EnsembleIndexer(1))
    super().__init__(enn.apply, enn.init, enn.indexer)
