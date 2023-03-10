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
"""Dataset transformations for generating OOD data."""

from typing import Dict, Tuple

from enn.datasets import base
import numpy as np
import tensorflow.compat.v2 as tf


def get_dataset_transform_from_type(
    dataset_type: base.OodVariant,
    ood_labels: np.ndarray,
) -> base.DatasetTransformer:
  """Returns a dataset transform from a type string."""
  if dataset_type == base.OodVariant.IN_DISTRIBUTION:
    return _make_ds_transform(ood_labels, ood_proportion=0.)
  elif dataset_type == base.OodVariant.OUT_DISTRIBUTION:
    return _make_ds_transform(ood_labels, ood_proportion=1.)
  elif dataset_type == base.OodVariant.WHOLE:
    return lambda x: x
  else:
    raise ValueError('Unsupported dataset type.')


def make_ood_transformers(
    num_classes: int,
    fraction_ood_classes: float = 0.2,
    ood_proportion_in_train: float = 0.001,
    seed: int = 321,
) -> Tuple[base.DatasetTransformer, Dict[
    str, base.DatasetTransformer]]:
  """Returns a tuple of ood transfomers for train and eval datasets.

  Args:
    num_classes: Number of possible classes in the train dataset.
    fraction_ood_classes: Fraction of classes considered as out of distribution.
      We will sample OOD classes randomly according to the fraction.
    ood_proportion_in_train: Fraction of data samples with OOD labels in the
      training dataset.
    seed: Random seed used to sample OOD labels and generate the training set.

  Returns:
    A tuple where the first element is an ood transfomer for train dataset and
    the second elemenet is a dictionary of ood transfomers for eval dataset.
  """
  ood_labels = sample_classes(
      num_classes=num_classes,
      num_samples=int(fraction_ood_classes * num_classes),
      seed=seed,
  )
  train_ds_transformer = _make_ds_transform(
      ood_labels=ood_labels,
      ood_proportion=ood_proportion_in_train,
      seed=seed,
  )
  eval_ds_transformers = dict()
  for dataset_type in base.OodVariant:
    eval_ds_transformers[dataset_type.value] = get_dataset_transform_from_type(
        dataset_type=dataset_type,
        ood_labels=ood_labels,
    )
  return (train_ds_transformer, eval_ds_transformers)


def sample_classes(num_classes: int, num_samples: int, seed: int) -> np.ndarray:
  """Sample a subset of size num_samples from [0, ..., num_classes - 1]."""
  rng = np.random.default_rng(seed)
  return rng.choice(range(num_classes), size=(num_samples,), replace=False)


def _make_ds_transform(
    ood_labels: np.ndarray,
    ood_proportion: float = 0.,
    seed: int = 0,
) -> base.DatasetTransformer:
  """Makes a TF dataset transformation that filters out certain labels.

  Args:
    ood_labels: An array of out-of-distribution labels.
    ood_proportion: Fraction of data samples with ood_labels in the new dataset.
    seed: Random seed used to generate the new dataset.

  Returns:
    A function that takes a TF dataset and returns a TF dataset.
  """
  assert (ood_proportion >= 0.) and (ood_proportion <= 1.)
  if not ood_labels.any():
    return lambda ds: ds

  def in_dist_predicate(batch: base.ArrayBatch) -> bool:
    return tf.reduce_all(tf.not_equal(batch.y, ood_labels))
  def out_dist_predicate(batch: base.ArrayBatch) -> bool:
    return not in_dist_predicate(batch)

  if ood_proportion == 0.:
    return lambda ds: ds.filter(in_dist_predicate)
  elif ood_proportion == 1:
    return lambda ds: ds.filter(out_dist_predicate)

  weights = (1. - ood_proportion, ood_proportion)

  def partial_filter(ds: tf.data.Dataset):
    ds_in_dist = ds.filter(in_dist_predicate)
    ds_out_dist = ds.filter(out_dist_predicate)
    return tf.data.Dataset.sample_from_datasets(
        datasets=[ds_in_dist, ds_out_dist],
        weights=weights,
        stop_on_empty_dataset=True,
        seed=seed,
    )

  return partial_filter
