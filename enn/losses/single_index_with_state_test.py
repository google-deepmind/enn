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
"""Tests for ENN single index losses with state."""

from typing import Dict, Text, Tuple

from absl.testing import absltest
from absl.testing import parameterized
from enn import base
from enn import networks
from enn import utils
from enn.losses import single_index_with_state
import haiku as hk
import jax
import numpy as np


class DummyLossFn(single_index_with_state.SingleIndexLossFnWithState):
  """A dummy loss fn that returns the normalized index as loss.

  It also returns a constant dummy metrics. It is meant to be used with an
  ensemble ENN. The index is assumed a uniform random integer in the interval
  [0, num_ensemble). The loss is normalized such that its mean is 1.
  """

  def __init__(self, num_ensemble: int, dummy_metrics: Dict[Text, int]):
    self._num_ensemble = num_ensemble
    self._dummy_metrics = dummy_metrics

  def __call__(
      self,
      apply: base.ApplyFn,
      params: hk.Params,
      state: hk.State,
      batch: base.Batch,
      index: base.Index,
  ) -> Tuple[base.Array, Tuple[hk.State, base.LossMetrics]]:
    """Computes a loss based on one batch of data and one index."""
    del apply, params, batch
    return ((2 * index + 1) / self._num_ensemble, (state, self._dummy_metrics))


class AvgSingleIndexLossTest(absltest.TestCase):

  def test_averaging(self):
    """Average of single loss fn should have same mean and smaller variance ."""

    num_ensemble = 10
    dummy_metrics = {'a': 0, 'b': 1}
    # A dummy loss fn that returns the normalized index as loss and two constant
    # metrics. Index is random but normalized such that its mean is 1.
    single_loss_fn = DummyLossFn(num_ensemble, dummy_metrics)

    num_index_samples = 100
    loss_fn = single_index_with_state.average_single_index_loss_with_state(
        single_loss_fn, num_index_samples)
    dummy_batch = base.Batch(np.ones([1, 1]), np.ones([1, 1]))
    enn = networks.MLPEnsembleMatchedPrior(
        output_sizes=[1],
        num_ensemble=num_ensemble,
        dummy_input=dummy_batch.x,
    )
    enn = utils.wrap_enn_as_enn_with_state(enn)

    loss, (unused_new_state, metrics) = loss_fn(
        enn=enn,
        params=dict(),
        state=dict(),
        batch=dummy_batch,
        key=jax.random.PRNGKey(0),
    )

    # Since the single loss has mean 1 the averaged loss also has mean 1 a
    # variance proportional to 1/np.sqrt(num_index_samples).
    self.assertAlmostEqual(
        loss,
        1.0,
        delta=5 / np.sqrt(num_index_samples),
        msg=f'Expected loss to be ~1.0 but it is {loss}')
    self.assertDictEqual(
        metrics, dummy_metrics,
        f'expected metrics to be {dummy_metrics} but it is {metrics}')


class XentLossTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._params = dict()
    cls._state = dict()
    cls._index = np.array([])

  @parameterized.parameters([2, 3, 5])
  def test_null_bootstrapping(self, num_classes: int):
    """Test computed loss is correct when there is no bootstrapping."""

    loss_fn = single_index_with_state.XentLossWithState(num_classes)

    batch_size = 4
    batch = base.Batch(
        x=np.expand_dims(np.arange(batch_size), 1),
        y=np.random.random_integers(0, num_classes - 1, size=(batch_size, 1)),
        data_index=np.expand_dims(np.arange(batch_size), 1),
    )

    # Test when apply always return a uniform distribution over labels
    apply = lambda p, s, x, i: (np.ones(shape=(x.shape[0], num_classes)), s)
    # Since the output is uniform the log loss is always log(1/num_classes).
    expected_loss = -np.log(1.0 / num_classes)
    loss, unused_metrics = loss_fn(
        apply=apply,
        params=self._params,
        state=self._state,
        batch=batch,
        index=self._index,
    )
    self.assertEqual(
        loss, expected_loss,
        (f'expected loss for uniform prediction is {expected_loss}, '
         f'but it is {loss}'))

    # Test when apply always predict the label to be 0.
    logits = np.array([100] + [0] * (num_classes - 1))
    apply = lambda p, s, x, i: (np.tile(logits, (x.shape[0], 1)), s)
    # Compute the expected log loss.
    expected_loss = (
        jax.nn.logsumexp(logits) - np.mean(batch.y == 0) * logits[0])
    loss, unused_metrics = loss_fn(
        apply=apply,
        params=self._params,
        state=self._state,
        batch=batch,
        index=self._index,
    )
    self.assertEqual(
        loss, expected_loss,
        (f'expected loss for predicting class 0 is {expected_loss}, '
         f'but it is {loss}'))

  @parameterized.parameters([2, 3, 5])
  def test_zero_bootstrapping(self, num_classes: int):
    """Test computed loss is zero when bootstrap weights are zero."""

    loss_fn = single_index_with_state.XentLossWithState(num_classes)
    batch_size = 4
    batch = base.Batch(
        x=np.expand_dims(np.arange(batch_size), 1),
        y=np.random.random_integers(0, num_classes - 1, size=(batch_size, 1)),
        data_index=np.expand_dims(np.arange(batch_size), 1),
        weights=np.zeros([batch_size, 1]),
    )

    # Test when apply always return a uniform distribution over labels
    apply = lambda p, s, x, i: (np.ones(shape=(x.shape[0], num_classes)), s)
    loss, unused_metrics = loss_fn(
        apply=apply,
        params=self._params,
        state=self._state,
        batch=batch,
        index=self._index,
    )
    self.assertEqual(
        loss, 0.0, ('expected loss with zero bootstrapping weights to be zero, '
                    f'but it is {loss}'))


if __name__ == '__main__':
  absltest.main()
