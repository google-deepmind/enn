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

"""Functions for 2D classification."""
from typing import Optional, Tuple

from enn import base as enn_base
from enn import supervised
from enn import utils
import haiku as hk
import jax
import numpy as np
import pandas as pd
import plotnine as gg
from sklearn import datasets


def make_dataset(num_sample: int = 10,
                 prob_swap: float = 0.,
                 seed: int = 0) -> enn_base.BatchIterator:
  """Make a 2 moons dataset with num_sample per class and prob_swap label."""
  x, y = datasets.make_moons(2 * num_sample, noise=0.1, random_state=seed)

  # Swap the labels for data with prob_swap
  swaps = np.random.RandomState(seed).binomial(1, prob_swap, len(y))
  swap_locs = np.where(swaps)[0]
  y[swap_locs] = 1 - y[swap_locs]

  return utils.make_batch_iterator(enn_base.Batch(x, y))


def make_dataframe(
    dataset: Optional[enn_base.BatchIterator] = None) -> pd.DataFrame:
  dataset = dataset or make_dataset()
  batch = next(dataset)
  vals = np.hstack([batch.x, batch.y])
  return pd.DataFrame(vals, columns=['x1', 'x2', 'label'])


def gen_2d_grid(plot_range: float) -> np.ndarray:
  """Generates a 2D grid for data in a certain_range."""
  data = []
  x_range = np.linspace(-plot_range, plot_range)
  for x1 in x_range:
    for x2 in x_range:
      data.append((x1, x2))
  return np.vstack(data)


def make_plot_data(experiment: supervised.BaseExperiment,
                   num_sample: int) -> pd.DataFrame:
  """Generate a classification plot with sampled predictions."""
  preds_x = gen_2d_grid(plot_range=3)

  data = []
  rng = hk.PRNGSequence(jax.random.PRNGKey(seed=0))
  for k in range(num_sample):
    net_out = experiment.predict(preds_x, key=next(rng))
    logits = utils.parse_net_output(net_out)
    preds_y = jax.nn.softmax(logits)
    data.append(pd.DataFrame({
        'x1': preds_x[:, 0], 'x2': preds_x[:, 1], 'label': preds_y[:, 1],
        'sample': k
    }))
  return pd.concat(data)


def make_sample_plot(plot_df: pd.DataFrame,
                     data_df: Optional[pd.DataFrame] = None):
  """Make a plot of 2D classification samples over dataset."""
  if data_df is None:
    data_df = make_dataframe()
  p = (gg.ggplot()
       + gg.aes('x1', 'x2', fill='label')
       + gg.geom_tile(data=plot_df, alpha=0.75)
       + gg.scale_fill_continuous(limits=[0, 1])
       + gg.geom_point(data=data_df,
                       colour='black', size=5, stroke=2)
       + gg.facet_wrap('sample', labeller='label_both')
       + gg.ggtitle('Posterior samples from ENN')
       + gg.theme(figure_size=(20, 14), panel_spacing=0.2))
  return p


def make_mean_plot(plot_df: pd.DataFrame,
                   data_df: Optional[pd.DataFrame] = None):
  """Make a plot of 2D classification of the mean of the samples."""
  mean_df = plot_df.groupby(['x1', 'x2'])['label'].mean().reset_index()
  if data_df is None:
    data_df = make_dataframe()
  p = (gg.ggplot()
       + gg.aes('x1', 'x2', fill='label')
       + gg.geom_tile(data=mean_df, alpha=0.75)
       + gg.scale_fill_continuous(limits=[0, 1])
       + gg.geom_point(data=data_df,
                       colour='black', size=5, stroke=2)
       + gg.ggtitle('Posterior mean from ENN')
       + gg.theme(figure_size=(12, 10), panel_spacing=0.2))
  return p


def make_mean_plot_data(
    experiment: supervised.BaseExperiment) -> Tuple[pd.DataFrame, pd.DataFrame]:
  plot_df = make_plot_data(experiment, num_sample=100)
  dataframe = make_dataframe(experiment.dataset)
  mean_df = plot_df.groupby(['x1', 'x2'])['label'].mean().reset_index()

  return mean_df, dataframe


def colab_plots(experiment: supervised.BaseExperiment):
  plot_df = make_plot_data(experiment, num_sample=100)
  dataframe = make_dataframe(experiment.dataset)
  make_mean_plot(plot_df, dataframe).draw()
  make_sample_plot(plot_df[plot_df['sample'] < 12],
                   dataframe).draw()
