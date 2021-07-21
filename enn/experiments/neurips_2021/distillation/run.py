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
"""Run a distillation ENN agent through the GP testbed."""

from absl import app
from absl import flags
from acme.utils import loggers
from enn import base as enn_base
from enn import data_noise
from enn import losses
from enn import networks
from enn.experiments.neurips_2021 import agents
from enn.experiments.neurips_2021 import base as testbed_base
from enn.experiments.neurips_2021 import load
from enn.experiments.neurips_2021.distillation import train_lib
import jax
from jax.config import config
import jax.numpy as jnp

# Double-precision in JAX helps with numerical stability
config.update('jax_enable_x64', True)

# GP configuration
flags.DEFINE_integer('input_dim', 1, 'Input dimension.')
flags.DEFINE_float('data_ratio', 1., 'Ratio of num_train to input_dim.')
flags.DEFINE_float('noise_std', 0.1, 'Additive noise standard deviation.')
flags.DEFINE_integer('seed', 1, 'Seed for testbed problem.')

# ENN training
flags.DEFINE_string('agent_name', 'distillation', 'Name of ENN agent.')
flags.DEFINE_integer('num_batches', 10_000, 'Number of batches to train')
flags.DEFINE_integer('num_ensemble', 10, 'Number of ensemble in ENN')
flags.DEFINE_float('prior_scale', 1, 'Prior scale for ENN.')
flags.DEFINE_float('noise_scale', 1, 'Noise scale for ENN.')
flags.DEFINE_float('l2_weight_decay', 0, 'Scale for l2 weight decay.')

FLAGS = flags.FLAGS


def make_enn(prior: testbed_base.PriorKnowledge) -> enn_base.EpistemicNetwork:
  enn = networks.make_ensemble_mlp_with_prior_enn(
      output_sizes=[50, 50, prior.num_classes],
      dummy_input=jnp.ones([prior.num_train, prior.input_dim]),
      num_ensemble=FLAGS.num_ensemble,
      prior_scale=FLAGS.prior_scale,
  )
  return train_lib.DistillRegressionMLP(enn)


def make_loss(prior: testbed_base.PriorKnowledge,
              enn: enn_base.EpistemicNetwork) -> enn_base.LossFn:
  noise_std = FLAGS.noise_scale * prior.noise_std
  noise_fn = data_noise.GaussianTargetNoise(enn, noise_std)
  single_loss = losses.add_data_noise(losses.L2Loss(), noise_fn)
  ave_loss = losses.average_single_index_loss(single_loss, FLAGS.num_ensemble)
  distill_loss = train_lib.DistillRegressionLoss(100, FLAGS.num_ensemble)
  return train_lib.combine_losses([ave_loss, distill_loss])


def extract_enn_sampler(
    agent: agents.VanillaEnnAgent) -> testbed_base.EpistemicSampler:
  """Extracts ENN sampler form the distillation component."""
  def enn_sampler(x: enn_base.Array, seed: int = 0) -> enn_base.Array:
    """Generate a random sample from posterior distribution at x."""
    net_out = agent.experiment.predict(x, seed)
    mean = net_out.extra['mean']
    std = jnp.sqrt(jnp.exp(net_out.extra['log_var']))
    noise = jax.random.normal(jax.random.PRNGKey(seed))
    return mean + noise * std
  return jax.jit(enn_sampler)


def main(_):
  # Load the appropriate testbed problem
  problem = load.regression_load(
      input_dim=FLAGS.input_dim,
      data_ratio=FLAGS.data_ratio,
      seed=FLAGS.seed,
      noise_std=FLAGS.noise_std,
  )

  agent_config = agents.VanillaEnnConfig(
      enn_ctor=make_enn,
      loss_ctor=make_loss,
      num_batches=FLAGS.num_batches,
      logger=loggers.make_default_logger('experiment', time_delta=0),
      seed=FLAGS.seed,
  )
  agent = agents.VanillaEnnAgent(agent_config)
  _ = agent(problem.train_data, problem.prior_knowledge)
  enn_sampler = extract_enn_sampler(agent)
  quality = problem.evaluate_quality(enn_sampler)
  print(quality.kl_estimate)

if __name__ == '__main__':
  app.run(main)
