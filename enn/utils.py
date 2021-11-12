# python3
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
from typing import Callable, Optional

from absl import flags
from enn import base
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from sklearn import datasets
import tensorflow as tf
import tensorflow_datasets as tfds

FLAGS = flags.FLAGS


def epistemic_network_from_module(
    enn_ctor: Callable[[], base.EpistemicModule],
    indexer: base.EpistemicIndexer,
) -> base.EpistemicNetwork:
  """Convert an Enn module to epistemic network with paired index."""

  def enn_fn(inputs: base.Array, index: base.Index) -> base.Output:
    return enn_ctor()(inputs, index)

  transformed = hk.without_apply_rng(hk.transform(enn_fn))
  return base.EpistemicNetwork(transformed.apply, transformed.init, indexer)


def wrap_transformed_as_enn(
    transformed: hk.Transformed) -> base.EpistemicNetwork:
  """Wraps a simple transformed function y = f(x) as an ENN."""
  return base.EpistemicNetwork(
      apply=lambda params, x, z: transformed.apply(params, x),
      init=lambda key, x, z: transformed.init(key, x),
      indexer=lambda key: key,
  )


def wrap_enn_as_enn_with_state(
    enn: base.EpistemicNetwork) -> base.EpistemicNetworkWithState:
  """Wraps a standard ENN as an ENN with a dummy network state."""
  def init(key, x, index):
    unused_state = {}
    return (enn.init(key, x, index), unused_state)
  def apply(params, unused_state, x, index):
    return (enn.apply(params, x, index), unused_state)
  return base.EpistemicNetworkWithState(
      apply=apply,
      init=init,
      indexer=enn.indexer,
  )


def make_centered_enn(enn: base.EpistemicNetwork,
                      x_train: base.Array) -> base.EpistemicNetwork:
  """Returns an ENN that centers input according to x_train."""
  assert x_train.ndim > 1  # need to include a batch dimension
  x_mean = jnp.mean(x_train, axis=0)
  x_std = jnp.std(x_train, axis=0)
  def centered_apply(params: hk.Params,
                     x: base.Array,
                     z: base.Index) -> base.Output:
    normalized_x = (x - x_mean) / (x_std + 1e-9)
    return enn.apply(params, normalized_x, z)
  return base.EpistemicNetwork(centered_apply, enn.init, enn.indexer)


def parse_net_output(net_out: base.Output) -> base.Array:
  """Convert potential dict of network outputs to scalar prediction value."""
  if isinstance(net_out, base.OutputWithPrior):
    return net_out.preds
  else:
    return net_out


def make_batch_indexer(indexer: base.EpistemicIndexer,
                       batch_size: int) -> base.EpistemicIndexer:
  """Batches an EpistemicIndexer to produce batch_size index samples."""
  fold_in = jax.vmap(jax.random.fold_in, in_axes=[None, 0])
  batch_array = jnp.arange(batch_size)

  def batch_indexer(key: base.RngKey) -> base.Index:
    batch_keys = fold_in(key, batch_array)
    return jax.vmap(indexer)(batch_keys)

  return batch_indexer


def _clean_batch_data(data: base.Batch) -> base.Batch:
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


def make_batch_iterator(data: base.Batch,
                        batch_size: Optional[int] = None,
                        seed: int = 0) -> base.BatchIterator:
  """Converts toy-like training data to batch_iterator for sgd training."""
  data = _clean_batch_data(data)
  n_data = len(data.y)
  if not batch_size:
    batch_size = n_data

  ds = tf.data.Dataset.from_tensor_slices(data).cache()
  ds = ds.shuffle(min(n_data, 50 * batch_size), seed=seed)
  ds = ds.repeat().batch(batch_size)

  return iter(tfds.as_numpy(ds))


def make_test_data(n_samples: int = 20) -> base.BatchIterator:
  """Generate a simple dataset suitable for classification or regression."""
  x, y = datasets.make_moons(n_samples, noise=0.1, random_state=0)
  return make_batch_iterator(base.Batch(x, y))
