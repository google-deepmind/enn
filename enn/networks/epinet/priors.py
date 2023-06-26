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
"""Functions for epinet and priors.

WARNING: NOT GOLD QUALITY YET - WORK IN PROGRESS.
"""

from typing import Optional, Sequence, Tuple

import chex
from enn import base
from enn.networks import base as networks_base
from enn.networks import einsum_mlp
from enn.networks import ensembles
from enn.networks import priors
from enn.networks.epinet import base as epinet_base
import haiku as hk
import jax
import jax.numpy as jnp


def combine_epinet_and_prior(
    epinet: epinet_base.EpinetWithState,
    prior_fn: priors.PriorFn,
    prior_scale: float = 1,
) -> epinet_base.EpinetWithState:
  """Combines epinet and prior_fn to give a new epinet."""
  def apply(
      params: hk.Params,  # Epinet parameters
      state: hk.State,  # Epinet state
      inputs: chex.Array,  # ENN inputs = x
      index: base.Index,  # ENN index = z
      hidden: chex.Array,  # Base net hiddens = phi(x)
  ) -> Tuple[networks_base.OutputWithPrior, hk.State]:
    epi_out, epi_state = epinet.apply(params, state, inputs, index, hidden)
    prior_out = prior_fn(inputs, index)
    combined_out = networks_base.OutputWithPrior(
        train=epi_out.train,
        prior=epi_out.prior + prior_out * prior_scale,
        extra=epi_out.extra
    )
    return combined_out, epi_state

  return epinet_base.EpinetWithState(apply, epinet.init, epinet.indexer)


def make_imagenet_mlp_prior(
    num_ensemble: int = 1,
    hidden_sizes: Sequence[int] = (50, 50),
    num_classes: int = 1000,
    seed: int = 23) -> priors.PriorFn:
  """Combining a few mlps as prior function."""
  # Note that this returns a callable function and no parameters are exposed.
  rng = hk.PRNGSequence(seed)
  output_sizes = list(hidden_sizes) + [num_classes,]
  def net_fn(x):
    # We need to transpose images to match the double-transpose-trick we have in
    # the ImageNet dataset loader (enn/datasets/imagenet.py).
    if jax.local_devices()[0].platform == 'tpu':
      x = jnp.transpose(x, (3, 0, 1, 2))  # HWCN -> NHWC
    x = hk.avg_pool(x, window_shape=10, strides=5, padding='VALID')
    model = einsum_mlp.EnsembleMLP(output_sizes, num_ensemble)
    return model(x)
  transformed = hk.without_apply_rng(hk.transform(net_fn))

  dummy_input = jnp.ones(shape=(1, 224, 224, 3))
  if jax.local_devices()[0].platform == 'tpu':
    dummy_input = jnp.transpose(dummy_input, (1, 2, 3, 0))  # NHWC -> HWCN
  params = transformed.init(next(rng), dummy_input)
  prior_fn = lambda x, z: jnp.dot(transformed.apply(params, x), z)
  return jax.jit(prior_fn)


def make_imagenet_conv_prior(
    num_ensemble: int = 1,
    num_classes: int = 1000,
    seed: int = 23,
    output_channels: Sequence[int] = (4, 8, 8),
    kernel_shapes: Sequence[int] = (10, 10, 3),
    strides: Sequence[int] = (5, 5, 2),
) -> priors.PriorFn:
  """Combining a few conv nets as prior function."""
  # Note that this returns a callable function and no parameters are exposed.
  rng = hk.PRNGSequence(seed)
  assert len(output_channels) == len(kernel_shapes) == len(strides)

  def conv_net(x):
    # We need to transpose images to match the double-transpose-trick we have in
    # the ImageNet dataset loader (enn/datasets/imagenet.py).
    if jax.local_devices()[0].platform == 'tpu':
      x = jnp.transpose(x, (3, 0, 1, 2))  # HWCN -> NHWC
    for channels, kernel_shape, stride in zip(output_channels,
                                              kernel_shapes,
                                              strides,):
      x = hk.Conv2D(output_channels=channels,
                    kernel_shape=kernel_shape,
                    stride=stride,
                    name='prior_conv')(x)
      x = jax.nn.relu(x)
    x = hk.Flatten()(x)
    return hk.nets.MLP([num_classes], name='prior')(x)

  transformed = hk.without_apply_rng(hk.transform(conv_net))
  ensemble = ensembles.Ensemble(model=transformed, num_ensemble=num_ensemble)

  dummy_index = ensemble.indexer(next(rng))
  dummy_input = jnp.ones(shape=(1, 224, 224, 3))
  if jax.local_devices()[0].platform == 'tpu':
    dummy_input = jnp.transpose(dummy_input, (1, 2, 3, 0))  # NHWC -> HWCN
  params = ensemble.init(next(rng), dummy_input, dummy_index)

  def prior_fn(x: chex.Array, z: chex.Array) -> chex.Array:
    out = [ensemble.apply(params, x, index) for index in range(num_ensemble)]
    out = jnp.stack(out, axis=-1)
    return jnp.dot(out, z)

  return jax.jit(prior_fn)


def make_cifar_conv_prior(
    num_ensemble: int = 1,
    num_classes: int = 10,
    seed: int = 23,
    output_channels: Sequence[int] = (4, 8, 4),
    kernel_shapes: Optional[Sequence[int]] = None,
) -> priors.PriorFn:
  """Combining a few conv nets as prior function."""
  rng = hk.PRNGSequence(seed)

  if kernel_shapes is None:
    kernel_shapes = [[5, 5]] * len(output_channels)
  assert len(output_channels) == len(kernel_shapes)

  def conv_net(x):
    x = jax.image.resize(x, [x.shape[0], 32, 32, 3], method='bilinear')
    for i, channels in enumerate(output_channels):
      x = hk.Conv2D(output_channels=channels,
                    kernel_shape=kernel_shapes[i],
                    stride=2, name='prior_conv')(x)
      x = jax.nn.relu(x)
    x = hk.Flatten()(x)
    return hk.nets.MLP([num_classes], name='prior')(x)

  transformed = hk.without_apply_rng(hk.transform(conv_net))
  ensemble = ensembles.Ensemble(model=transformed, num_ensemble=num_ensemble)

  dummy_index = ensemble.indexer(next(rng))
  dummy_input = jnp.ones(shape=(4, 32, 32, 3))
  params = ensemble.init(next(rng), dummy_input, dummy_index)

  def prior_fn(x: chex.Array, z: chex.Array) -> chex.Array:
    out = [ensemble.apply(params, x, index) for index in range(num_ensemble)]
    out = jnp.stack(out, axis=-1)
    return jnp.dot(out, z)

  return jax.jit(prior_fn)
