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
"""Utility functions for working with datasets."""

from typing import Dict, Sequence

import chex
from enn import base
from enn.datasets import base as ds_base
import jax
import tensorflow.compat.v2 as tf


def change_ds_dict_to_enn_batch(
    ds_dict: Dict[str, chex.Array]) -> base.Batch:
  """Changes a dictionary of (image, label) to an enn batch."""
  assert 'image' in ds_dict
  assert 'label' in ds_dict
  return base.Batch(x=ds_dict['image'], y=ds_dict['label'])


def add_data_index_to_dataset(ds: tf.data.Dataset) -> tf.data.Dataset:
  """Adds integer data_index into the batch dictionary."""
  ds = ds.enumerate()
  return ds.map(_add_data_index)


def slice_dataset_to_batches(
    dataset: tf.data.Dataset,
    total_batch_size: int,
    data_parallelism: bool,
) -> tf.data.Dataset:
  """Slices the data of a dataset into batches.

  Args:
    dataset: a tf.data dataset.
    total_batch_size: the total batch size over all devices.
    data_parallelism: a boolean specifying whether to add a leading axis for the
      number of devices. If true, data batches have shape
      (number_devices, total_batch_size / number_devices, ...); otherwise, the
      have shape of (total_batch_size, ...).

  Returns:
    a tf.data dataset sliced into batches.
  """
  if data_parallelism:
    per_device_batch_size = get_per_device_batch_size(total_batch_size)
    dataset = dataset.batch(per_device_batch_size, drop_remainder=True)
    dataset = dataset.batch(jax.local_device_count(), drop_remainder=True)
  else:
    dataset = dataset.batch(total_batch_size, drop_remainder=True)
  return dataset


def get_per_device_batch_size(total_batch_size: int) -> int:
  """Calculates the batch size per device based on total batch size."""
  num_devices = jax.device_count()
  per_device_batch_size, ragged = divmod(total_batch_size, num_devices)

  if ragged:
    raise ValueError(
        f'Global batch size {total_batch_size} must be divisible by the '
        f'total number of devices {num_devices}')
  return per_device_batch_size


def _add_data_index(data_index: int, batch: base.Batch) -> base.Batch:
  """Adds data_index into the batch."""
  return base.Batch(x=batch.x, y=batch.y, data_index=data_index)


class OverrideTrainDataset(ds_base.DatasetWithTransform):
  """Overrides the train dataset with a replacement dataset."""

  def __init__(self,
               original_dataset: ds_base.DatasetWithTransform,
               new_dataset: ds_base.DatasetWithTransform):
    assert original_dataset.num_classes == new_dataset.num_classes
    self.original_dataset = original_dataset
    self.new_dataset = new_dataset
    self.train_ds_transformer = original_dataset.train_ds_transformer
    self.eval_ds_transformers = original_dataset.eval_ds_transformers

  @property
  def num_classes(self) -> int:
    return self.original_dataset.num_classes

  @property
  def eval_input_shape(self) -> Sequence[int]:
    return self.original_dataset.eval_input_shape

  def train_dataset(self) -> ds_base.DatasetGenerator:
    return self.new_dataset.train_dataset()

  def eval_datasets(self) -> Dict[str, ds_base.DatasetGenerator]:
    return self.original_dataset.eval_datasets()
