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
"""ImageNet dataset with typical pre-processing."""
import dataclasses
import enum
import itertools as it
from typing import Dict, Optional, Sequence, Tuple

from enn import base as enn_base
from enn.datasets import base as ds_base
from enn.datasets import utils as ds_utils
import jax
import jax.numpy as jnp
from jaxline import utils
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


MEAN_RGB = (0.485 * 255, 0.456 * 255, 0.406 * 255)
STDDEV_RGB = (0.229 * 255, 0.224 * 255, 0.225 * 255)


class Split(enum.Enum):
  """Imagenet dataset split."""
  TRAIN = 1271167
  TRAIN_AND_VALID = 1281167
  VALID = 10000
  TEST = 50000

  @property
  def num_examples(self):
    return self.value

MAX_NUM_TRAIN = Split.TRAIN_AND_VALID.num_examples


@dataclasses.dataclass
class Imagenet(ds_base.DatasetWithTransform):
  """Imagenet as jaxline dataset."""
  train_batch: int = 128
  eval_batch: int = 100
  dataset_seed: int = 0
  enable_double_transpose: bool = True
  fake_data: bool = False
  num_train: int = Split.TRAIN_AND_VALID.num_examples
  train_ds_transformer: ds_base.DatasetTransformer = lambda x: x
  eval_ds_transformers: Dict[
      str, ds_base.DatasetTransformer] = ds_base.EVAL_TRANSFORMERS_DEFAULT

  @property
  def num_classes(self) -> int:
    return 1000

  def train_dataset(self) -> ds_base.DatasetGenerator:
    """Returns the train dataset."""
    def build_train_input() -> ds_base.DatasetGenerator:
      """See base class."""
      num_devices = jax.device_count()
      global_batch_size = self.train_batch
      per_device_batch_size, ragged = divmod(global_batch_size, num_devices)

      if ragged:
        raise ValueError(
            f'Global batch size {global_batch_size} must be divisible by '
            f'num devices {num_devices}')
      # double-transpose-trick is only needed on TPU.
      should_transpose_images = (
          self.enable_double_transpose and
          jax.local_devices()[0].platform == 'tpu')

      return load(
          Split.TRAIN_AND_VALID,
          is_training=True,
          transpose=should_transpose_images,
          batch_dims=[jax.local_device_count(), per_device_batch_size],
          fake_data=self.fake_data,
          seed=self.dataset_seed,
          ds_transform=self.train_ds_transformer,
          num_examples=self.num_train)

    train_input = utils.py_prefetch(build_train_input)
    return utils.double_buffer_on_gpu(train_input)

  def eval_datasets(self) -> Dict[str, ds_base.DatasetGenerator]:
    """Returns the evaluation dataset."""

    def build_eval_dataset(
        eval_ds_transformer: ds_base.DatasetTransformer
    ) -> ds_base.DatasetGenerator:
      # double-transpose-trick is only needed on TPU.
      should_transpose_images = (
          self.enable_double_transpose and
          jax.local_devices()[0].platform == 'tpu')

      return load(
          Split.TEST,
          is_training=False,
          transpose=should_transpose_images,
          batch_dims=[self.eval_batch],
          fake_data=self.fake_data,
          seed=self.dataset_seed,
          ds_transform=eval_ds_transformer,)

    return {
        dataset_type: build_eval_dataset(transformer) for
        dataset_type, transformer in self.eval_ds_transformers.items()
    }


def load(
    split: Split,
    *,
    is_training: bool,
    batch_dims: Sequence[int],
    dtype: jnp.dtype = jnp.float32,
    transpose: bool = False,
    fake_data: bool = False,
    image_size: Tuple[int, int] = (224, 224),
    seed: Optional[int] = None,
    ds_transform: ds_base.DatasetTransformer = lambda x: x,
    num_examples: Optional[int] = None,
) -> ds_base.DatasetGenerator:
  """Loads the given split of the dataset."""
  start, end = _shard(
      split,
      shard_index=jax.process_index(),
      num_shards=jax.process_count(),
      num_examples=num_examples)

  # Run deterministically if rng is not None
  if seed is not None:
    rng = tf.random.create_rng_state(seed, 'threefry')
    rng = tf.random.experimental.stateless_fold_in(
        tf.cast(rng, tf.int64), start)
    # Splitting the rng - one is used as seed for shuffling, the other is
    # used as seed for random crop
    rngs = tf.random.experimental.stateless_split(rng, 2)

  if fake_data:
    images = np.zeros(tuple(batch_dims) + image_size + (3,), dtype=dtype)
    labels = np.zeros(tuple(batch_dims), dtype=np.int32)
    if transpose:
      axes = tuple(range(images.ndim))
      axes = axes[:-4] + axes[-3:] + (axes[-4],)  # NHWC -> HWCN
      images = np.transpose(images, axes)
    batch = enn_base.Batch(x=images, y=labels)
    yield from it.repeat(batch, end - start)
    return

  total_batch_size = np.prod(batch_dims)

  tfds_split = tfds.core.ReadInstruction(
      _to_tfds_split(split), from_=start, to=end, unit='abs')

  ds = tfds.load(
      'imagenet2012:5.*.*',
      split=tfds_split,
      decoders={'image': tfds.decode.SkipDecoding()})
  ds = ds.map(ds_utils.change_ds_dict_to_enn_batch)
  ds = ds_utils.add_data_index_to_dataset(ds)

  options = tf.data.Options()
  options.experimental_threading.private_threadpool_size = 48
  options.experimental_threading.max_intra_op_parallelism = 1
  options.experimental_optimization.map_parallelization = True

  if seed is None and is_training:
    options.experimental_deterministic = False
  ds = ds.with_options(options)

  if is_training:
    if jax.process_count() > 1:
      # Only cache if we are reading a subset of the dataset.
      ds = ds.cache()
    ds = ds.repeat()
    seed_shuffle = tf.cast(rngs[0][0],
                           tf.int64) if seed is not None else 0
    ds = ds.shuffle(buffer_size=10 * total_batch_size, seed=seed_shuffle)

  else:
    if split.num_examples % total_batch_size != 0:
      raise ValueError(f'Test/valid must be divisible by {total_batch_size}')

  def _preprocess_fn(batch: enn_base.Batch,
                     example_rng: Optional[tf.Tensor] = None) -> enn_base.Batch:
    image = _preprocess_image(batch.x, is_training, image_size,
                              example_rng)
    return ds_utils.update_x_in_batch(batch=batch, x=image)

  def _preprocess_with_per_example_rng(
      ds: tf.data.Dataset, *, rng: Optional[np.ndarray]) -> tf.data.Dataset:

    def _fn(example_index: int, batch: enn_base.Batch) -> enn_base.Batch:
      example_rng = None
      if rng is not None:
        example_index = tf.cast(example_index, tf.int32)
        example_rng = tf.random.experimental.stateless_fold_in(
            tf.cast(rng, tf.int64), example_index)
      processed = _preprocess_fn(batch, example_rng)
      return processed

    return ds.enumerate().map(
        _fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  if seed is None:
    ds = ds.map(
        _preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  else:
    rng_process = tf.cast(rngs[1], tf.int64)
    ds = _preprocess_with_per_example_rng(ds, rng=rng_process)

  # TODO(author2): This transform needs to come after processing.
  ds = ds_transform(ds)

  def transpose_fn(batch: enn_base.Batch) -> enn_base.Batch:
    # We use double-transpose-trick to improve performance for TPUs. Note
    # that this (typically) requires a matching HWCN->NHWC transpose in your
    # model code. The compiler cannot make this optimization for us since our
    # data pipeline and model are compiled separately.
    transposed_x = tf.transpose(batch.x, (1, 2, 3, 0))
    return ds_utils.update_x_in_batch(batch=batch, x=transposed_x)

  def cast_fn(batch: enn_base.Batch) -> enn_base.Batch:
    x = tf.cast(batch.x, tf.dtypes.as_dtype(dtype))
    return ds_utils.update_x_in_batch(batch=batch, x=x)

  for i, batch_size in enumerate(reversed(batch_dims)):
    ds = ds.batch(batch_size, drop_remainder=True)
    if i == 0:
      if transpose:
        ds = ds.map(transpose_fn)  # NHWC -> HWCN
      # NOTE: You may be tempted to move the casting earlier on in the pipeline,
      # but for bf16 some operations will end up silently placed on the TPU and
      # this causes stalls while TF and JAX battle for the accelerator.
      ds = ds.map(cast_fn)

  ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

  yield from tfds.as_numpy(ds)


def _to_tfds_split(split: Split) -> tfds.Split:
  """Returns the TFDS split appropriately sharded."""
  # NOTE: Imagenet did not release labels for the test split used in the
  # competition, so it has been typical at DeepMind to consider the VALID
  # split the TEST split and to reserve 10k images from TRAIN for VALID.
  if split in (Split.TRAIN, Split.TRAIN_AND_VALID, Split.VALID):
    return tfds.Split.TRAIN
  else:
    assert split == Split.TEST
    return tfds.Split.VALIDATION


def _shard(split: Split,
           shard_index: int,
           num_shards: int,
           num_examples: Optional[int] = None) -> Tuple[int, int]:
  """Returns [start, end) for the given shard index."""
  assert shard_index < num_shards
  if num_examples:
    assert num_examples >= 1
    num_examples = min(num_examples, split.num_examples)
  else:
    num_examples = split.num_examples
  arange = np.arange(num_examples)
  shard_range = np.array_split(arange, num_shards)[shard_index]
  start, end = shard_range[0], (shard_range[-1] + 1)
  if split == Split.TRAIN:
    # Note that our TRAIN=TFDS_TRAIN[10000:] and VALID=TFDS_TRAIN[:10000].
    offset = Split.VALID.num_examples
    start += offset
    end += offset
  return start, end


def _preprocess_image(
    image_bytes: tf.Tensor,
    is_training: bool,
    image_size: Sequence[int],
    rng: Optional[tf.Tensor] = None,
) -> tf.Tensor:
  """Returns processed and resized images."""
  if is_training:
    if rng is not None:
      rngs = tf.random.experimental.stateless_split(rng, 2)
      image = _decode_and_random_crop(image_bytes, rngs[0], image_size)
      image = tf.image.stateless_random_flip_left_right(image, rngs[1])
    else:
      image = _decode_and_random_crop(image_bytes, None, image_size)
      image = tf.image.random_flip_left_right(image, None)
  else:
    image = _decode_and_center_crop(image_bytes, image_size=image_size)
  assert image.dtype == tf.uint8
  # NOTE: Bicubic resize (1) casts uint8 to float32 and (2) resizes without
  # clamping overshoots. This means values returned will be outside the range
  # [0.0, 255.0] (e.g. we have observed outputs in the range [-51.1, 336.6]).
  image = tf.image.resize(image, image_size, tf.image.ResizeMethod.BICUBIC)
  image = _normalize_image(image)
  return image


def _normalize_image(image: tf.Tensor) -> tf.Tensor:
  """Normalize the image to zero mean and unit variance."""
  image -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=image.dtype)
  image /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=image.dtype)
  return image


def _distorted_bounding_box_crop(
    image_bytes: tf.Tensor,
    *,
    jpeg_shape: tf.Tensor,
    bbox: tf.Tensor,
    seed: Optional[tf.Tensor],
    min_object_covered: float,
    aspect_ratio_range: Tuple[float, float],
    area_range: Tuple[float, float],
    max_attempts: int,
) -> tf.Tensor:
  """Generates cropped_image using one of the bboxes randomly distorted."""
  kwargs = {
      'image_size': jpeg_shape,
      'bounding_boxes': bbox,
      'min_object_covered': min_object_covered,
      'aspect_ratio_range': aspect_ratio_range,
      'area_range': area_range,
      'max_attempts': max_attempts,
      'use_image_if_no_bounding_boxes': True
  }
  if seed is not None:
    bbox_begin, bbox_size, _ = tf.image.stateless_sample_distorted_bounding_box(
        seed=seed, **kwargs)
  else:
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(**kwargs)

  # Crop the image to the specified bounding box.
  offset_y, offset_x, _ = tf.unstack(bbox_begin)
  target_height, target_width, _ = tf.unstack(bbox_size)
  crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
  image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
  return image


def _decode_and_random_crop(
    image_bytes: tf.Tensor,
    seed: Optional[tf.Tensor],
    image_size: Sequence[int] = (224, 224),
) -> tf.Tensor:
  """Make a random crop of 224."""
  jpeg_shape = tf.image.extract_jpeg_shape(image_bytes)
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  image = _distorted_bounding_box_crop(
      image_bytes,
      jpeg_shape=jpeg_shape,
      bbox=bbox,
      seed=seed,
      min_object_covered=0.1,
      aspect_ratio_range=(3 / 4, 4 / 3),
      area_range=(0.08, 1.0),
      max_attempts=10)
  if tf.reduce_all(tf.equal(jpeg_shape, tf.shape(image))):
    # If the random crop failed fall back to center crop.
    image = _decode_and_center_crop(image_bytes, jpeg_shape, image_size)
  return image


def _decode_and_center_crop(
    image_bytes: tf.Tensor,
    jpeg_shape: Optional[tf.Tensor] = None,
    image_size: Sequence[int] = (224, 224),
) -> tf.Tensor:
  """Crops to center of image with padding then scales."""
  if jpeg_shape is None:
    jpeg_shape = tf.image.extract_jpeg_shape(image_bytes)
  image_height = jpeg_shape[0]
  image_width = jpeg_shape[1]

  # Pad the image with at least 32px on the short edge and take a
  # crop that maintains aspect ratio.
  scale = tf.minimum(
      tf.cast(image_height, tf.float32) / (image_size[0] + 32),
      tf.cast(image_width, tf.float32) / (image_size[1] + 32))
  padded_center_crop_height = tf.cast(scale * image_size[0], tf.int32)
  padded_center_crop_width = tf.cast(scale * image_size[1], tf.int32)
  offset_height = ((image_height - padded_center_crop_height) + 1) // 2
  offset_width = ((image_width - padded_center_crop_width) + 1) // 2
  crop_window = tf.stack([
      offset_height, offset_width, padded_center_crop_height,
      padded_center_crop_width
  ])
  image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
  return image
