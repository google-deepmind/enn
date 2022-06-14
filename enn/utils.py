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
"""Utility functions."""
from typing import Callable, Optional, Tuple

from absl import flags
from enn import base_legacy
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from sklearn import datasets
import tensorflow as tf
import tensorflow_datasets as tfds

FLAGS = flags.FLAGS


def epistemic_network_from_module(
    enn_ctor: Callable[[], base_legacy.EpistemicModule],
    indexer: base_legacy.EpistemicIndexer,
) -> base_legacy.EpistemicNetwork:
  """Convert an Enn module to epistemic network with paired index."""

  def enn_fn(inputs: base_legacy.Array,
             index: base_legacy.Index) -> base_legacy.Output:
    return enn_ctor()(inputs, index)

  transformed = hk.without_apply_rng(hk.transform(enn_fn))
  return base_legacy.EpistemicNetwork(transformed.apply, transformed.init,
                                      indexer)


def wrap_transformed_as_enn(
    transformed: hk.Transformed) -> base_legacy.EpistemicNetwork:
  """Wraps a simple transformed function y = f(x) as an ENN."""
  return base_legacy.EpistemicNetwork(
      apply=lambda params, x, z: transformed.apply(params, x),
      init=lambda key, x, z: transformed.init(key, x),
      indexer=lambda key: key,
  )


def wrap_enn_as_enn_with_state(
    enn: base_legacy.EpistemicNetwork) -> base_legacy.EpistemicNetworkWithState:
  """Wraps a standard ENN as an ENN with a dummy network state."""

  def init(
      key: base_legacy.RngKey,
      inputs: base_legacy.Array,
      index: base_legacy.Array,
  ) -> Tuple[hk.Params, hk.State]:
    return (enn.init(key, inputs, index), {})

  def apply(
      params: hk.Params,
      unused_state: hk.State,
      inputs: base_legacy.Array,
      index: base_legacy.Array,
  ) -> Tuple[base_legacy.Output, hk.State]:
    return (enn.apply(params, inputs, index), {})

  return base_legacy.EpistemicNetworkWithState(
      apply=apply,
      init=init,
      indexer=enn.indexer,
  )


def wrap_enn_with_state_as_enn(
    enn: base_legacy.EpistemicNetworkWithState,
    constant_state: Optional[hk.State] = None,
) -> base_legacy.EpistemicNetwork:
  """Passes a dummy state to ENN with state as an ENN."""
  if constant_state is None:
    constant_state = {}

  def init(key: base_legacy.RngKey, x: base_legacy.Array,
           z: base_legacy.Index) -> hk.Params:
    params, unused_state = enn.init(key, x, z)
    return params

  def apply(params: hk.Params, x: base_legacy.Array,
            z: base_legacy.Index) -> base_legacy.Output:
    output, unused_state = enn.apply(params, constant_state, x, z)
    return output

  return base_legacy.EpistemicNetwork(
      apply=apply,
      init=init,
      indexer=enn.indexer,
  )


def scale_enn_output(
    enn: base_legacy.EpistemicNetworkWithState,
    scale: float,
) -> base_legacy.EpistemicNetworkWithState:
  """Returns an ENN with output scaled by a scaling factor."""
  def scaled_apply(
      params: hk.Params, state: hk.State, inputs: base_legacy.Array,
      index: base_legacy.Index) -> Tuple[base_legacy.Output, hk.State]:
    out, state = enn.apply(params, state, inputs, index)
    if isinstance(out, base_legacy.OutputWithPrior):
      scaled_out = base_legacy.OutputWithPrior(
          train=out.train * scale,
          prior=out.prior * scale,
          extra=out.extra,
      )
    else:
      scaled_out = out * scale
    return scaled_out, state

  return base_legacy.EpistemicNetworkWithState(
      apply=scaled_apply,
      init=enn.init,
      indexer=enn.indexer,
  )


def make_centered_enn(
    enn: base_legacy.EpistemicNetwork,
    x_train: base_legacy.Array) -> base_legacy.EpistemicNetwork:
  """Returns an ENN that centers input according to x_train."""
  assert x_train.ndim > 1  # need to include a batch dimension
  x_mean = jnp.mean(x_train, axis=0)
  x_std = jnp.std(x_train, axis=0)
  def centered_apply(params: hk.Params, x: base_legacy.Array,
                     z: base_legacy.Index) -> base_legacy.Output:
    normalized_x = (x - x_mean) / (x_std + 1e-9)
    return enn.apply(params, normalized_x, z)

  return base_legacy.EpistemicNetwork(centered_apply, enn.init, enn.indexer)


def parse_net_output(net_out: base_legacy.Output) -> base_legacy.Array:
  """Convert network output to scalar prediction value."""
  if isinstance(net_out, base_legacy.OutputWithPrior):
    return net_out.preds
  else:
    return net_out


def parse_to_output_with_prior(
    net_out: base_legacy.Output) -> base_legacy.OutputWithPrior:
  """Convert network output to base.OutputWithPrior."""
  if isinstance(net_out, base_legacy.OutputWithPrior):
    return net_out
  else:
    return base_legacy.OutputWithPrior(
        train=net_out, prior=jnp.zeros_like(net_out))


def make_batch_indexer(indexer: base_legacy.EpistemicIndexer,
                       batch_size: int) -> base_legacy.EpistemicIndexer:
  """Batches an EpistemicIndexer to produce batch_size index samples."""
  fold_in = jax.vmap(jax.random.fold_in, in_axes=[None, 0])
  batch_array = jnp.arange(batch_size)

  def batch_indexer(key: base_legacy.RngKey) -> base_legacy.Index:
    batch_keys = fold_in(key, batch_array)
    return jax.vmap(indexer)(batch_keys)

  return batch_indexer


def _clean_batch_data(data: base_legacy.Batch) -> base_legacy.Batch:
  """Checks some of the common shape/index issues for dummy data.."""
  # Make sure that the data has a separate batch dimension
  if data.y.ndim == 1:
    data = data._replace(y=data.y[:, None])

  # Data index to identify each instance
  if data.data_index is None:
    data = data._replace(data_index=np.arange(len(data.y))[:, None])

  # Weights to say how much each data.point is work
  if data.weights is None:
    data = data._replace(weights=np.ones(len(data.y))[:, None])
  return data


def make_batch_iterator(data: base_legacy.Batch,
                        batch_size: Optional[int] = None,
                        seed: int = 0) -> base_legacy.BatchIterator:
  """Converts toy-like training data to batch_iterator for sgd training."""
  data = _clean_batch_data(data)
  n_data = len(data.y)
  if not batch_size:
    batch_size = n_data

  ds = tf.data.Dataset.from_tensor_slices(data).cache()
  ds = ds.shuffle(min(n_data, 50 * batch_size), seed=seed)
  ds = ds.repeat().batch(batch_size)

  return iter(tfds.as_numpy(ds))


def make_test_data(n_samples: int = 20) -> base_legacy.BatchIterator:
  """Generate a simple dataset suitable for classification or regression."""
  x, y = datasets.make_moons(n_samples, noise=0.1, random_state=0)
  return make_batch_iterator(base_legacy.Batch(x, y))
