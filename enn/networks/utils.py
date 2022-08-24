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
"""Utility functions for networks."""
from typing import Callable, Optional, Tuple

import chex
from enn import base
from enn.networks import base as networks_base
import haiku as hk
import jax.numpy as jnp


def parse_net_output(net_out: networks_base.Output) -> chex.Array:
  """Convert network output to scalar prediction value."""
  if isinstance(net_out, networks_base.OutputWithPrior):
    return net_out.preds
  else:
    return net_out


def parse_to_output_with_prior(
    net_out: networks_base.Output) -> networks_base.OutputWithPrior:
  """Convert network output to networks_base.OutputWithPrior."""
  if isinstance(net_out, networks_base.OutputWithPrior):
    return net_out
  else:
    return networks_base.OutputWithPrior(
        train=net_out, prior=jnp.zeros_like(net_out))


def epistemic_network_from_module(
    enn_ctor: Callable[[], networks_base.EpistemicModule],
    indexer: base.EpistemicIndexer,
) -> networks_base.EnnArray:
  """Convert an Enn module to epistemic network with paired index."""

  def enn_fn(inputs: chex.Array,
             index: base.Index) -> networks_base.Output:
    return enn_ctor()(inputs, index)

  transformed = hk.without_apply_rng(hk.transform_with_state(enn_fn))
  return networks_base.EnnArray(transformed.apply, transformed.init, indexer)


# TODO(author4): Sort out issues with importing the function in networks/__init__.
def wrap_net_fn_as_enn(
    net_fn: Callable[[base.Input], base.Output],  # pre-transformed
) -> base.EpistemicNetwork[base.Input, base.Output]:
  """Wrap a pre-transformed function as an ENN with dummy index.

  Args:
    net_fn: A pre-transformed net function y = f(x). We assume that the network
      doesn't use rng during apply internally.
  Returns:
    An ENN that wraps around f(x) with a dummy indexer.
  """
  transformed = hk.without_apply_rng(hk.transform_with_state(net_fn))

  def apply(
      params: hk.Params,
      state: hk.State,
      inputs: base.Input,
      index: base.Index,
  ) -> Tuple[base.Output, hk.State]:
    del index
    return transformed.apply(params, state, inputs)

  return base.EpistemicNetwork[base.Input, base.Output](
      apply=apply,
      init=lambda k, x, z: transformed.init(k, x),
      indexer=lambda k: k,
  )


def wrap_transformed_as_enn_no_state(
    transformed: hk.Transformed) -> networks_base.EnnNoState:
  """Wraps a simple transformed function y = f(x) as an ENN."""
  return networks_base.EnnNoState(
      apply=lambda params, x, z: transformed.apply(params, x),
      init=lambda key, x, z: transformed.init(key, x),
      indexer=lambda key: key,
  )


def wrap_transformed_as_enn(
    transformed: hk.Transformed
) -> networks_base.EnnArray:
  """Wraps a simple transformed function y = f(x) as an ENN."""
  apply = lambda params, x, z: transformed.apply(params, x)
  apply = wrap_apply_no_state_as_apply(apply)
  init = lambda key, x, z: transformed.init(key, x)
  init = wrap_init_no_state_as_init(init)
  return networks_base.EnnArray(
      apply=apply,
      init=init,
      indexer=lambda key: key,
  )


def wrap_enn_no_state_as_enn(
    enn: networks_base.EnnNoState
) -> networks_base.EnnArray:
  """Wraps a standard ENN as an ENN with a dummy network state."""

  return networks_base.EnnArray(
      apply=wrap_apply_no_state_as_apply(enn.apply),
      init=wrap_init_no_state_as_init(enn.init),
      indexer=enn.indexer,
  )


def wrap_enn_as_enn_no_state(
    enn: networks_base.EnnArray,
    constant_state: Optional[hk.State] = None,
) -> networks_base.EnnNoState:
  """Passes a dummy state to ENN with state as an ENN."""
  if constant_state is None:
    constant_state = {}

  def init(key: chex.PRNGKey, x: chex.Array, z: base.Index) -> hk.Params:
    params, unused_state = enn.init(key, x, z)
    return params

  def apply(
      params: hk.Params, x: chex.Array, z: base.Index) -> networks_base.Output:
    output, unused_state = enn.apply(params, constant_state, x, z)
    return output

  return networks_base.EnnNoState(
      apply=apply,
      init=init,
      indexer=enn.indexer,
  )


def wrap_apply_no_state_as_apply(
    apply: networks_base.ApplyNoState,) -> networks_base.ApplyArray:
  """Wraps a legacy enn apply as an apply for enn with state."""
  def new_apply(
      params: hk.Params,
      unused_state: hk.State,
      inputs: chex.Array,
      index: base.Index,
  ) -> Tuple[networks_base.Output, hk.State]:
    return (apply(params, inputs, index), {})
  return new_apply


def wrap_init_no_state_as_init(
    init: networks_base.InitNoState) -> networks_base.InitArray:
  """Wraps a legacy enn init as an init for enn with state."""

  def new_init(
      key: chex.PRNGKey,
      inputs: chex.Array,
      index: base.Index,
  ) -> Tuple[hk.Params, hk.State]:
    return (init(key, inputs, index), {})
  return new_init


def scale_enn_output(
    enn: networks_base.EnnArray,
    scale: float,
) -> networks_base.EnnArray:
  """Returns an ENN with output scaled by a scaling factor."""
  def scaled_apply(
      params: hk.Params, state: hk.State, inputs: chex.Array,
      index: base.Index) -> Tuple[networks_base.Output, hk.State]:
    out, state = enn.apply(params, state, inputs, index)
    if isinstance(out, networks_base.OutputWithPrior):
      scaled_out = networks_base.OutputWithPrior(
          train=out.train * scale,
          prior=out.prior * scale,
          extra=out.extra,
      )
    else:
      scaled_out = out * scale
    return scaled_out, state

  return networks_base.EnnArray(
      apply=scaled_apply,
      init=enn.init,
      indexer=enn.indexer,
  )


def make_centered_enn_no_state(
    enn: networks_base.EnnNoState,
    x_train: chex.Array) -> networks_base.EnnNoState:
  """Returns an ENN that centers input according to x_train."""
  assert x_train.ndim > 1  # need to include a batch dimension
  x_mean = jnp.mean(x_train, axis=0)
  x_std = jnp.std(x_train, axis=0)
  def centered_apply(params: hk.Params, x: chex.Array,
                     z: base.Index) -> networks_base.Output:
    normalized_x = (x - x_mean) / (x_std + 1e-9)
    return enn.apply(params, normalized_x, z)

  return networks_base.EnnNoState(centered_apply, enn.init, enn.indexer)


def make_centered_enn(
    enn: networks_base.EnnArray,
    x_train: chex.Array) -> networks_base.EnnArray:
  """Returns an ENN that centers input according to x_train."""
  assert x_train.ndim > 1  # need to include a batch dimension
  x_mean = jnp.mean(x_train, axis=0)
  x_std = jnp.std(x_train, axis=0)
  def centered_apply(params: hk.Params, state: hk.State, x: chex.Array,
                     z: base.Index) -> networks_base.Output:
    normalized_x = (x - x_mean) / (x_std + 1e-9)
    return enn.apply(params, state, normalized_x, z)

  return networks_base.EnnArray(centered_apply, enn.init, enn.indexer)
