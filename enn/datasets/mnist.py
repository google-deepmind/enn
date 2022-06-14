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
"""Data loader for MNIST dataset."""
import dataclasses
import enum
import functools
from typing import Dict, Sequence

from enn import base_legacy as enn_base
from enn.datasets import base as ds_base
from enn.datasets import utils as ds_utils
import jax
from jaxline import utils
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

AUTOTUNE = tf.data.experimental.AUTOTUNE


class Split(enum.Enum):
  """Mnist dataset split."""
  TRAIN = 50000
  TEST = 10000

  @property
  def num_examples(self):
    return self.value


@dataclasses.dataclass
class Mnist(ds_base.DatasetWithTransform):
  """MNIST as a jaxline dataset."""
  train_batch: int = 128
  eval_batch: int = 100
  normalization_mode: str = 'standard'
  train_ds_transformer: ds_base.DatasetTransformer = lambda x: x
  eval_ds_transformers: Dict[
      str, ds_base.DatasetTransformer] = ds_base.EVAL_TRANSFORMERS_DEFAULT

  @property
  def num_classes(self) -> int:
    return 10

  @property
  def eval_input_shape(self) -> Sequence[int]:
    return (28, 28, 1)

  def train_dataset(self,) -> ds_base.DatasetGenerator:
    """Returns the train dataset."""
    def build_train_input():
      num_devices = jax.device_count()
      total_batch_size = self.train_batch
      per_device_batch_size, ragged = divmod(total_batch_size, num_devices)

      if ragged:
        raise ValueError(
            f'Global batch size {total_batch_size} must be divisible by the '
            f'total number of devices {num_devices}')

      ds = tfds.load(name='mnist', split='train')
      ds = ds.map(ds_utils.change_ds_dict_to_enn_batch)
      ds = self.train_ds_transformer(ds)
      ds = ds_utils.add_data_index_to_dataset(ds)
      ds = ds.shard(jax.process_count(), jax.process_index())
      # Shuffle before repeat ensures all examples seen in an epoch.
      # https://www.tensorflow.org/guide/data_performance#repeat_and_shuffle.
      ds = ds.shuffle(buffer_size=10_000)
      ds = ds.repeat()
      train_preprocess = functools.partial(
          preprocess_batch, normalization_mode=self.normalization_mode)
      ds = ds.map(train_preprocess, num_parallel_calls=AUTOTUNE)
      ds = ds.batch(per_device_batch_size, drop_remainder=True)
      ds = ds.batch(jax.local_device_count(), drop_remainder=True)
      ds = ds.prefetch(AUTOTUNE)
      return iter(tfds.as_numpy(ds))

    train_input = utils.py_prefetch(build_train_input)
    return utils.double_buffer_on_gpu(train_input)

  def eval_datasets(self,) -> Dict[str, ds_base.DatasetGenerator]:
    """Returns the evaluation dataset."""

    def build_eval_dataset(
        eval_ds_transformer: ds_base.DatasetTransformer
    ) -> ds_base.DatasetGenerator:
      ds = tfds.load(name='mnist', split='test')
      ds = ds.map(ds_utils.change_ds_dict_to_enn_batch)
      ds = ds_utils.add_data_index_to_dataset(ds)
      # Preprocess
      eval_preprocess = functools.partial(
          preprocess_batch, normalization_mode=self.normalization_mode)
      ds = ds.map(eval_preprocess, num_parallel_calls=AUTOTUNE)
      # Apply evaluation transformer
      ds = eval_ds_transformer(ds)
      ds = ds.batch(self.eval_batch, drop_remainder=True)
      ds = ds.prefetch(AUTOTUNE)
      return iter(tfds.as_numpy(ds))

    return {
        dataset_type: build_eval_dataset(transformer) for
        dataset_type, transformer in self.eval_ds_transformers.items()
    }


def preprocess_batch(batch: enn_base.Batch,
                     normalization_mode: str) -> enn_base.Batch:
  """Pre-processing module."""
  images = batch.x

  images = tf.image.convert_image_dtype(images, tf.float32)

  if normalization_mode == 'standard':
    images = tf.image.per_image_standardization(images)
  elif normalization_mode == 'identity':
    pass
  else:
    raise ValueError(
        'Normalization mode should be one among custom, standard or identity.'
    )

  return batch._replace(x=images)
