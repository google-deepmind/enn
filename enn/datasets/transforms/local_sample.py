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
"""Transformation to sample from dataset with local structure.

This is used to generate evaluation batches that are of the (kappa, N) format.
"""

from typing import Callable, Optional

from enn import base_legacy as enn_base
from enn.datasets import base as ds_base
import tensorflow.compat.v2 as tf


AUTOTUNE = tf.data.experimental.AUTOTUNE
PerturbFn = Callable[[enn_base.Batch], enn_base.Batch]


def make_repeat_sample_transform(
    num_repeat: int,
    perturb_fn: PerturbFn,
    limit_data: Optional[int] = None,
) -> ds_base.DatasetTransformer:
  """Alters Dataset for batches with num_repeat items and perturb_fn applied.

  This function will alter the dataset so that each original entry is then
  sequentially replaced by num_repeat copies of that entry, but with the
  perturb_fn appplied to that entry. This is useful for highlighting the
  importance of epistemic/aleatoric uncertainty.

  Args:
    num_repeat: number of repeated entries.
    perturb_fn: function to be applied to each entry.
    limit_data: optionally limit the final size of the dataset.

  Returns:
    dataset transformer.
  """

  def repeat(batch: enn_base.Batch) -> enn_base.Batch:
    repeated_x = tf.stack([batch.x] * num_repeat)
    repeated_y = tf.stack([batch.y] * num_repeat)
    return enn_base.Batch(x=repeated_x, y=repeated_y)

  def transform(ds: tf.data.Dataset) -> tf.data.Dataset:
    ds = ds.map(repeat).unbatch()
    if limit_data:
      ds = ds.take(limit_data)
    ds = ds.map(perturb_fn, num_parallel_calls=AUTOTUNE)
    return ds

  return transform


def make_dyadic_transform(
    num_repeat: int,
    limit_data: Optional[int] = None,
    crop_offset: int = 4,
    flip: bool = True,
) -> ds_base.DatasetTransformer:
  """Defines settings perturbing images with random crop/flip."""
  def perturb_fn(batch: enn_base.Batch) -> enn_base.Batch:
    images = batch.x
    if crop_offset > 0:
      image_height, image_width, image_depth = images.shape
      assert image_height > crop_offset
      assert image_width > crop_offset
      cropped_image_shape = (image_height - crop_offset,
                             image_width - crop_offset, image_depth)
      images = tf.image.random_crop(images, cropped_image_shape)
      # Resizing cropped image to its original shape. Without resizing, cropped
      # image may not be passed to the neural network.
      images = tf.image.resize(images, (image_height, image_width),
                               tf.image.ResizeMethod.BICUBIC)
    if flip:
      images = tf.image.random_flip_left_right(images)
    return batch.replace(x=images)

  return make_repeat_sample_transform(num_repeat, perturb_fn, limit_data)
