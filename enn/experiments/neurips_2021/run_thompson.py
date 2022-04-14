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
"""Example running an ENN on Thompson bandit task."""

from absl import app
from absl import flags
from enn.experiments.neurips_2021 import agent_factories
from enn.experiments.neurips_2021 import thompson
from jax.config import config

# Double-precision in JAX helps with numerical stability
config.update('jax_enable_x64', True)

# GP configuration
flags.DEFINE_integer('input_dim', 1, 'Input dimension.')
flags.DEFINE_float('action_ratio', 10., 'Ratio of actions to input_dim.')
flags.DEFINE_float('noise_std', 0.1, 'Additive noise standard deviation.')
flags.DEFINE_integer('seed', 1, 'Seed for testbed problem.')
flags.DEFINE_integer('num_steps', 100_000, 'Number of steps to run experiment.')


# ENN agent
flags.DEFINE_integer('agent_id', 0, 'Which agent id')
flags.DEFINE_enum('agent', 'all',
                  ['all', 'ensemble', 'dropout', 'hypermodel', 'bbb'],
                  'Which agent family.')

FLAGS = flags.FLAGS


def main(_):
  # Form the appropriate agent for training
  agent_config = agent_factories.load_agent_config(FLAGS.agent_id, FLAGS.agent)

  # Evaluate performance in Thompson sampling
  experiment = thompson.ThompsonEnnBandit(
      agent_config,
      input_dim=FLAGS.input_dim,
      noise_std=FLAGS.noise_std,
      num_actions=int(FLAGS.action_ratio * FLAGS.input_dim),
      seed=FLAGS.seed,
  )
  log_freq = int(FLAGS.num_steps / 100)
  if log_freq == 0:
    log_freq = 1
  experiment.run(FLAGS.num_steps, log_freq)
  print(f'total_regret={experiment.total_regret}')

if __name__ == '__main__':
  app.run(main)
