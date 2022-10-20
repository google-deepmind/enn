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
"""Data loader for CIFAR10/100 dataset."""
import dataclasses
import enum
import functools
from typing import Dict, Sequence

from enn import base
from enn.datasets import base as ds_base
from enn.datasets import utils as ds_utils
import jax
from jaxline import utils
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

AUTOTUNE = tf.data.experimental.AUTOTUNE


class CifarVariant(enum.Enum):
  """Variants of Cifar daaset."""
  CIFAR10 = 10
  CIFAR100 = 100

  @property
  def num_classes(self) -> int:
    return self.value

  @property
  def tfds_name(self) -> str:
    return f'cifar{self.value}'


class Split(enum.Enum):
  """Cifar dataset split."""
  TRAIN = 50000
  TEST = 10000

  @property
  def num_examples(self) -> int:
    return self.value


@dataclasses.dataclass
class Cifar(ds_base.DatasetWithTransform):
  """Cifar dataset."""
  cifar_variant: CifarVariant = CifarVariant.CIFAR10
  train_batch: int = 128
  eval_batch: int = 100
  normalization_mode: str = 'custom'
  random_flip: bool = True
  random_crop: bool = True
  cutout: bool = False
  keep_image_size: bool = False
  num_train: int = 50_000
  train_ds_transformer: ds_base.DatasetTransformer = lambda x: x
  eval_ds_transformers: Dict[
      str, ds_base.DatasetTransformer] = ds_base.EVAL_TRANSFORMERS_DEFAULT
  # Whether to add a leading axis of number of devices to the batches. If true,
  # data batches have shape (number_devices, batch_size / number_devices, ...);
  # otherwise, they have shape of (batch_size, ...).
  train_data_parallelism: bool = True  # Slices the train data into devices.
  eval_data_parallelism: bool = False

  @property
  def num_classes(self) -> int:
    return self.cifar_variant.num_classes

  @property
  def eval_input_shape(self) -> Sequence[int]:
    return (32, 32, 3)

  def train_dataset(self,) -> ds_base.DatasetGenerator:
    """Returns the train dataset."""
    def build_train_input():
      ds = tfds.load(
          name=self.cifar_variant.tfds_name,
          split=f'train[:{self.num_train}]',
      )
      ds = ds.map(ds_utils.change_ds_dict_to_enn_batch)
      ds = ds_utils.add_data_index_to_dataset(ds)
      ds = self.train_ds_transformer(ds)
      ds = ds.shard(jax.process_count(), jax.process_index())
      # Shuffle before repeat ensures all examples seen in an epoch.
      # https://www.tensorflow.org/guide/data_performance#repeat_and_shuffle
      ds = ds.shuffle(buffer_size=10_000)
      ds = ds.repeat()
      train_preprocess = functools.partial(
          preprocess_batch,
          normalization_mode=self.normalization_mode,
          random_flip=self.random_flip,
          random_crop=self.random_crop,
          cutout=self.cutout,
          keep_image_size=self.keep_image_size,
          is_training=True,
      )
      ds = ds.map(train_preprocess, num_parallel_calls=AUTOTUNE)
      ds = ds_utils.slice_dataset_to_batches(
          dataset=ds,
          total_batch_size=self.train_batch,
          data_parallelism=self.train_data_parallelism,
      )
      ds = ds.prefetch(AUTOTUNE)
      return iter(tfds.as_numpy(ds))

    train_input = utils.py_prefetch(build_train_input)
    return utils.double_buffer_on_gpu(train_input)

  def eval_datasets(self,) -> Dict[str, ds_base.DatasetGenerator]:
    """Returns the evaluation dataset."""

    def build_eval_dataset(
        eval_ds_transformer: ds_base.DatasetTransformer
    ) -> ds_base.DatasetGenerator:
      ds = tfds.load(name=self.cifar_variant.tfds_name, split='test')
      ds = ds.map(ds_utils.change_ds_dict_to_enn_batch)
      ds = ds_utils.add_data_index_to_dataset(ds)
      ds = ds.shard(jax.process_count(), jax.process_index())
      # Preprocess
      eval_preprocess = functools.partial(
          preprocess_batch,
          normalization_mode=self.normalization_mode,
          random_flip=self.random_flip,
          random_crop=self.random_crop,
          cutout=self.cutout,
          keep_image_size=self.keep_image_size,
          is_training=False,
      )
      ds = ds.map(eval_preprocess, num_parallel_calls=AUTOTUNE)
      # Apply evaluation transformer
      ds = eval_ds_transformer(ds)
      ds = ds_utils.slice_dataset_to_batches(
          dataset=ds,
          total_batch_size=self.eval_batch,
          data_parallelism=self.eval_data_parallelism,
      )
      ds = ds.prefetch(AUTOTUNE)
      return iter(tfds.as_numpy(ds))

    return {
        dataset_type: build_eval_dataset(transformer) for
        dataset_type, transformer in self.eval_ds_transformers.items()
    }


@dataclasses.dataclass
class Cifar10(Cifar):
  """Cifar10 as jaxline dataset."""
  cifar_variant: CifarVariant = CifarVariant.CIFAR10


@dataclasses.dataclass
class Cifar100(Cifar):
  """Cifar100 as jaxline dataset."""
  cifar_variant: CifarVariant = CifarVariant.CIFAR100
  cutout: bool = True
  keep_image_size: bool = True


def preprocess_batch(batch: base.Batch,
                     normalization_mode: str,
                     random_crop: bool,
                     random_flip: bool,
                     cutout: bool,
                     keep_image_size: bool,
                     is_training: bool = False) -> base.Batch:
  """Pre-processing module."""
  images = batch.x
  images = tf.image.convert_image_dtype(images, tf.float32)
  if normalization_mode != 'custom':
    images = images * 2. - 1.
    tf.assert_less_equal(tf.math.reduce_max(images), 1.)
    tf.assert_greater_equal(tf.math.reduce_min(images), -1.)

  if normalization_mode == 'custom':
    means = [0.49139968, 0.48215841, 0.44653091]
    stds = [0.24703223, 0.24348513, 0.26158784]
    images = (images - means) / stds
  elif normalization_mode == 'standard':
    images = tf.image.per_image_standardization(images)
  elif normalization_mode == 'identity':
    pass
  else:
    raise ValueError(
        'Normalization mode should be one among custom, standard or identity.'
    )

  # Transformations that are valid only in training.
  if is_training:
    images = tf.reshape(images, (32, 32, 3))
    if random_crop:
      if keep_image_size:
        image_shape = tf.shape(images)
        images = tf.image.resize_with_crop_or_pad(images, image_shape[0] + 4,
                                                  image_shape[1] + 4)
        images = tf.image.random_crop(images, (32, 32, 3))
      else:
        images = tf.image.random_crop(images, (24, 24, 3))
    if random_flip:
      images = tf.image.random_flip_left_right(images)
    if cutout:
      images = _cutout_single_image(
          probability=0.5, cutout_size=16, image=images)

  return batch._replace(x=images)


def _cutout_single_image(
    probability: float, cutout_size: int, image: np.ndarray) -> np.ndarray:
  """Cutout function."""
  tf.Assert(
      tf.less(cutout_size, image.shape[0]),
      [cutout_size, image.shape[0]])
  tf.Assert(
      tf.less(cutout_size, image.shape[1]),
      [cutout_size, image.shape[1]])
  x_range = image.shape[0] - cutout_size + 1
  y_range = image.shape[1] - cutout_size + 1
  x_before = tf.random.uniform([], minval=0, maxval=x_range, dtype=tf.int32)
  y_before = tf.random.uniform([], minval=0, maxval=y_range, dtype=tf.int32)
  x_after = image.shape[0] - x_before - cutout_size
  y_after = image.shape[1] - y_before - cutout_size
  cutout_square = tf.zeros([cutout_size, cutout_size, 3])
  mask = tf.pad(
      cutout_square, [[x_before, x_after], [y_before, y_after], [0, 0]],
      constant_values=1.0)
  pred = tf.less(tf.random.uniform([], minval=0.0, maxval=1.0), probability)
  return tf.cond(pred, lambda: mask * image, lambda: image)
