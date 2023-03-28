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
# TODO(author7): move this file to enn.datasets.
import dataclasses
from typing import Optional

from absl import flags
import chex
from enn import base
from enn.datasets import base as ds_base
import jax
import jax.numpy as jnp
import numpy as np
from sklearn import datasets
import tensorflow as tf
import tensorflow_datasets as tfds

FLAGS = flags.FLAGS


def make_batch_indexer(indexer: base.EpistemicIndexer,
                       batch_size: int) -> base.EpistemicIndexer:
  """Batches an EpistemicIndexer to produce batch_size index samples."""
  fold_in = jax.vmap(jax.random.fold_in, in_axes=[None, 0])
  batch_array = jnp.arange(batch_size)

  def batch_indexer(key: chex.PRNGKey) -> base.Index:
    batch_keys = fold_in(key, batch_array)
    return jax.vmap(indexer)(batch_keys)

  return batch_indexer


def _clean_batch_data(data: ds_base.ArrayBatch) -> ds_base.ArrayBatch:
  """Checks some of the common shape/index issues for dummy data.."""
  # Make sure that the data has a separate batch dimension
  if data.y.ndim == 1:
    data = dataclasses.replace(data, y=data.y[:, None])

  # Data index to identify each instance
  if data.data_index is None:
    data = dataclasses.replace(data, data_index=np.arange(len(data.y))[:, None])

  # Weights to say how much each data.point is work
  if data.weights is None:
    data = dataclasses.replace(data, weights=np.ones(len(data.y))[:, None])
  return data


def make_batch_iterator(data: ds_base.ArrayBatch,
                        batch_size: Optional[int] = None,
                        seed: int = 0) -> ds_base.ArrayBatchIterator:
  """Converts toy-like training data to batch_iterator for sgd training."""
  data = _clean_batch_data(data)
  n_data = len(data.y)
  if not batch_size:
    batch_size = n_data

  ds = tf.data.Dataset.from_tensor_slices(data).cache()
  ds = ds.shuffle(min(n_data, 50 * batch_size), seed=seed)
  ds = ds.repeat().batch(batch_size)

  return iter(tfds.as_numpy(ds))


def make_test_data(n_samples: int = 20) -> ds_base.ArrayBatchIterator:
  """Generate a simple dataset suitable for classification or regression."""
  x, y = datasets.make_moons(n_samples, noise=0.1, random_state=0)
  return make_batch_iterator(ds_base.ArrayBatch(x=x, y=y))
