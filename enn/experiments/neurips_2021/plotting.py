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

"""Functions to sanity-check output in 1D plots."""

import chex
from enn.experiments.neurips_2021 import base as testbed_base
from enn.experiments.neurips_2021 import testbed
import haiku as hk
import numpy as np
import pandas as pd
import plotnine as gg

# Setting plot theme
gg.theme_set(gg.theme_bw(base_size=16, base_family='serif'))
gg.theme_update(figure_size=(12, 8), panel_spacing=0.5)


def sanity_1d(true_model: testbed_base.TestbedProblem,
              enn_sampler: testbed_base.EpistemicSampler) -> gg.ggplot:
  """Sanity check to plot 1D representation of the GP testbed output."""
  if hasattr(true_model, 'problem'):
    true_model = true_model.problem  # Removing logging wrappers
  if not hasattr(true_model, 'data_sampler'):
    return gg.ggplot()
  gp_model = true_model.data_sampler
  return plot_1d_regression(gp_model, enn_sampler)


def _gen_samples(enn_sampler: testbed_base.EpistemicSampler,
                 x: chex.Array,
                 num_samples: int) -> pd.DataFrame:
  """Generate posterior samples at x (not implemented for all posterior)."""
  data = []
  rng = hk.PRNGSequence(0)
  for seed in range(num_samples):
    net_out = enn_sampler(x, next(rng))
    y = net_out[:, 0]
    data.append(pd.DataFrame({'x': x[:, 0], 'y': y, 'seed': seed}))
  return pd.concat(data)


def plot_1d_regression(gp_model: testbed.GPRegression,
                       enn_sampler: testbed_base.EpistemicSampler,
                       num_samples: int = 100) -> gg.ggplot:
  """Plots 1D regression with confidence intervals."""
  # Training data
  train_data = gp_model.train_data
  df = pd.DataFrame({'x': train_data.x[:, 0], 'y': train_data.y[:, 0]})
  # Posterior data
  posterior_df = pd.DataFrame({
      'x': gp_model.x_test[:, 0],
      'y': gp_model.test_mean[:, 0],
      'std': np.sqrt(np.diag(gp_model.test_cov)),
  })
  posterior_df['method'] = 'gp'
  # ENN data
  sample_df = _gen_samples(enn_sampler, gp_model.x_test, num_samples)
  enn_df = sample_df.groupby('x')['y'].agg([np.mean, np.std]).reset_index()
  enn_df = enn_df.rename({'mean': 'y'}, axis=1)
  enn_df['method'] = 'enn'
  p = (gg.ggplot(pd.concat([posterior_df, enn_df]))
       + gg.aes(x='x', y='y', ymin='y-std', ymax='y+std', group='method')
       + gg.geom_ribbon(gg.aes(fill='method'), alpha=0.25)
       + gg.geom_line(gg.aes(colour='method'), size=2)
       + gg.geom_point(gg.aes(x='x', y='y'), data=df, size=4, inherit_aes=False)
       + gg.scale_colour_manual(['#e41a1c', '#377eb8'])
       + gg.scale_fill_manual(['#e41a1c', '#377eb8'])
      )
  return p
