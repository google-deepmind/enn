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
"""Tests for datasets."""

from typing import Dict, Tuple

from absl.testing import absltest
from absl.testing import parameterized
import chex
from enn import base as enn_base
from enn import datasets


def _make_eval_transformers(
    num_classes: int) -> Dict[str, datasets.DatasetTransformer]:
  _, eval_transformers = datasets.make_ood_transformers(
      num_classes=num_classes,
      fraction_ood_classes=0.2,
      ood_proportion_in_train=0.001,
      seed=321,
  )
  return eval_transformers


class DatasetsTest(parameterized.TestCase):

  @parameterized.parameters(
      {
          'dataset': datasets.Mnist(train_batch=2),
          'image_shape': (28, 28, 1),
      },
      {
          'dataset': datasets.Cifar10(train_batch=2),
          'image_shape': (24, 24, 3),
      },
      {
          'dataset': datasets.Cifar100(train_batch=2),
          'image_shape': (32, 32, 3),
      },
  )
  def test_train_dataset_no_ood(
      self,
      dataset: datasets.Dataset,
      image_shape: Tuple[int],):
    """Tests that one batch of train dataset can be loaded and it has the correct shape."""
    train_data_generator = dataset.train_dataset()
    train_data = next(train_data_generator)

    self.assertIsInstance(train_data, enn_base.Batch)

    chex.assert_shape(train_data.x, (1, 2) + image_shape)
    chex.assert_shape(train_data.y, (1, 2))
    chex.assert_shape(train_data.data_index, (1, 2))

  @parameterized.parameters(
      {'dataset': datasets.Mnist(eval_batch=2), 'image_shape': (28, 28, 1),},
      {'dataset': datasets.Cifar100(eval_batch=2), 'image_shape': (32, 32, 3),},
  )
  def test_eval_dataset_no_ood(
      self,
      dataset: datasets.Dataset,
      image_shape: Tuple[int],
  ):
    """Tests that one batch of eval dataset can be loaded and it has the correct shape."""
    eval_datasets = dataset.eval_datasets()

    # `eval_datasets` should have only one key
    self.assertLen(eval_datasets.keys(), 1)
    self.assertIn(datasets.OodVariant.WHOLE.value, eval_datasets)
    eval_data_generator = eval_datasets[datasets.OodVariant.WHOLE.value]
    eval_data = next(eval_data_generator)

    self.assertIsInstance(eval_data, enn_base.Batch)

    chex.assert_shape(eval_data.x, (2,) + image_shape)
    chex.assert_shape(eval_data.y, (2,))
    chex.assert_shape(eval_data.data_index, (2,))

  @parameterized.parameters(
      {
          'dataset':
              datasets.Mnist(
                  eval_batch=2,
                  eval_ds_transformers=_make_eval_transformers(num_classes=10),
              ),
          'image_shape': (28, 28, 1)
      },
      {
          'dataset':
              datasets.Cifar10(
                  eval_batch=2,
                  eval_ds_transformers=_make_eval_transformers(num_classes=10),
              ),
          'image_shape': (32, 32, 3)
      },
      {
          'dataset':
              datasets.Cifar100(
                  eval_batch=2,
                  eval_ds_transformers=_make_eval_transformers(num_classes=10),
              ),
          'image_shape': (32, 32, 3)
      },
  )
  def test_eval_dataset_ood(
      self,
      dataset: datasets.Dataset,
      image_shape: Tuple[int],
  ):
    """Tests that correct eval datsets are generated when we have ood."""

    eval_datasets = dataset.eval_datasets()

    # `eval_datasets` should have three keys
    self.assertLen(eval_datasets.keys(), 3)
    for ood_variant in [
        datasets.OodVariant.WHOLE.value,
        datasets.OodVariant.IN_DISTRIBUTION.value,
        datasets.OodVariant.OUT_DISTRIBUTION.value,
    ]:
      self.assertIn(ood_variant, eval_datasets)
      eval_data_generator = eval_datasets[ood_variant]
      eval_data = next(eval_data_generator)

      self.assertIsInstance(eval_data, enn_base.Batch)

      chex.assert_shape(eval_data.x, (2,) + image_shape)
      chex.assert_shape(eval_data.y, (2,))
      chex.assert_shape(eval_data.data_index, (2,))


if __name__ == '__main__':
  absltest.main()
