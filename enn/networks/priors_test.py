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

"""Tests for enn.priors."""
from typing import List

from absl.testing import absltest
from absl.testing import parameterized
from enn import supervised
from enn.networks import priors
from enn.networks import utils as network_utils
import haiku as hk
import jax


class PriorsTest(parameterized.TestCase):

  @parameterized.parameters([[[50, 50], True], [[20, 20], False]])
  def test_mlp_prior_module(self, hiddens: List[int], regression: bool):
    """Test MLP with prior from hk.Module."""
    test_experiment = supervised.make_test_experiment(regression)
    def net_fn(x):
      net = priors.NetworkWithAdditivePrior(
          net=hk.nets.MLP(hiddens + [test_experiment.num_outputs]),
          prior_net=hk.nets.MLP(hiddens + [test_experiment.num_outputs]),
          prior_scale=1.,
      )
      return net(x)

    transformed = hk.without_apply_rng(hk.transform(net_fn))
    enn = network_utils.wrap_transformed_as_enn(transformed)
    experiment = test_experiment.experiment_ctor(enn)
    experiment.train(10)

  @parameterized.parameters([[[50, 50], True], [[20, 20], False]])
  def test_mlp_prior_transformed(self, hiddens: List[int], regression: bool):
    """Test MLP with prior from EpistemicNetwork."""
    test_experiment = supervised.make_test_experiment(regression)
    def net_fn(x):
      net = hk.nets.MLP(hiddens + [test_experiment.num_outputs])
      return net(x)
    transformed = hk.without_apply_rng(hk.transform(net_fn))
    train_enn = network_utils.wrap_transformed_as_enn(transformed)

    prior_params = transformed.init(
        jax.random.PRNGKey(0), test_experiment.dummy_input)
    prior_fn = lambda x, z: transformed.apply(prior_params, x)
    enn = priors.EnnStateWithAdditivePrior(
        enn=train_enn,
        prior_fn=prior_fn,
        prior_scale=1.,
    )
    experiment = test_experiment.experiment_ctor(enn)
    experiment.train(10)

  @parameterized.parameters([
      [1, 3, 10, 10],
      [2, 5, 1, 10],
      [5, 1, 5, 10],
      [5, 5, 5, 1],
  ])
  def test_random_gp_forward(
      self, input_dim: int, output_dim: int, num_feat: int, batch_size: int):
    """Test random gp can forward data correcly."""
    rng = hk.PRNGSequence(0)
    fake_data = jax.random.normal(next(rng), shape=[batch_size, input_dim])
    gp_instance = priors.make_random_feat_gp(
        input_dim, output_dim, num_feat, next(rng))
    output = gp_instance(fake_data)
    assert output.shape == (batch_size, output_dim)

  @parameterized.parameters([[[50, 50], True], [[20, 20], False]])
  def test_get_random_mlp_prior_fn(self, hiddens: List[int], regression: bool):
    """Test MLP with prior from EpistemicNetwork."""
    test_experiment = supervised.make_test_experiment(regression)
    output_sizes = hiddens + [test_experiment.num_outputs]
    def net_fn(x):
      net = hk.nets.MLP(output_sizes)
      return net(x)
    transformed = hk.without_apply_rng(hk.transform(net_fn))
    train_enn = network_utils.wrap_transformed_as_enn(transformed)

    dummy_x = test_experiment.dummy_input
    dummy_z = train_enn.indexer(jax.random.PRNGKey(0))
    rng_seq = hk.PRNGSequence(0)
    prior_fn = priors.get_random_mlp_with_index(
        x_sample=dummy_x, z_sample=dummy_z, rng=next(rng_seq),
        prior_output_sizes=output_sizes)

    enn = priors.EnnStateWithAdditivePrior(
        enn=train_enn,
        prior_fn=prior_fn,
        prior_scale=1.,
    )
    experiment = test_experiment.experiment_ctor(enn)
    experiment.train(10)


if __name__ == '__main__':
  absltest.main()
