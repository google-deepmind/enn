# pylint: disable=g-bad-file-header
# Copyright 2023 DeepMind Technologies Limited. All Rights Reserved.
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
"""A JAX implementation of BERT."""

import typing as tp

import chex
from enn import base as enn_base
from enn import networks
from enn.networks.bert import base
import haiku as hk
import jax
import jax.numpy as jnp


# BERT layer norm uses
# github.com/google-research/tf-slim/blob/master/tf_slim/layers/layers.py#L2346
TF_LAYERNORM_EPSILON = 1e-12


def make_bert_enn(
    bert_config: base.BertConfig,
    is_training: bool,
) -> base.BertEnn:
  """Makes the BERT model as an ENN with state."""

  def net_fn(inputs: base.BertInput) -> networks.OutputWithPrior:
    """Forwards the network (no index)."""
    hidden_drop = bert_config.hidden_dropout_prob if is_training else 0.
    att_drop = bert_config.attention_probs_dropout_prob if is_training else 0.
    bert_model = BERT(
        vocab_size=bert_config.vocab_size,
        hidden_size=bert_config.hidden_size,
        num_hidden_layers=bert_config.num_hidden_layers,
        num_attention_heads=bert_config.num_attention_heads,
        intermediate_size=bert_config.intermediate_size,
        hidden_dropout_prob=hidden_drop,
        attention_probs_dropout_prob=att_drop,
        max_position_embeddings=bert_config.max_position_embeddings,
        type_vocab_size=bert_config.type_vocab_size,
        initializer_range=bert_config.initializer_range,
    )

    # Embed and summarize the sequence.
    return bert_model(  # pytype: disable=wrong-arg-types  # jax-devicearray
        input_ids=inputs.token_ids,
        token_type_ids=inputs.segment_ids,
        input_mask=inputs.input_mask.astype(jnp.int32),
        is_training=is_training,
    )

  # Transformed has the rng input, which we need to change --> index.
  transformed = hk.transform_with_state(net_fn)
  def apply(
      params: hk.Params,
      state: hk.State,
      inputs: base.BertInput,
      index: enn_base.Index,  # BERT operates with an RNG-key index.
  ) -> tp.Tuple[networks.OutputWithPrior, hk.State]:
    key = index
    return transformed.apply(params, state, key, inputs)
  def init(rng_key: chex.PRNGKey,
           inputs: base.BertInput,
           index: enn_base.Index) -> tp.Tuple[hk.Params, hk.State]:
    del index  # rng_key is duplicated in this case.
    return transformed.init(rng_key, inputs)

  return base.BertEnn(apply, init, networks.PrngIndexer())


class BERT(hk.Module):
  """BERT as a Haiku module.

  This is the Haiku module of the BERT model implemented in tensorflow here:
  https://github.com/google-research/bert/blob/master/modeling.py#L107
  """

  def __init__(
      self,
      vocab_size: int,
      hidden_size: int,
      num_hidden_layers: int,
      num_attention_heads: int,
      intermediate_size: int,
      hidden_dropout_prob: float,
      attention_probs_dropout_prob: float,
      max_position_embeddings: int,
      type_vocab_size: int,
      initializer_range: float,
      name: str = 'BERT',
  ):
    super().__init__(name=name)
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.intermediate_size = intermediate_size
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.max_position_embeddings = max_position_embeddings
    self.type_vocab_size = type_vocab_size
    self.initializer_range = initializer_range
    self.size_per_head = hidden_size // num_attention_heads

  def _bert_layer(
      self,
      layer_input: jnp.DeviceArray,
      layer_index: int,
      input_mask: jnp.DeviceArray,
      is_training: bool,
  ) -> jnp.DeviceArray:
    """Forward pass of a single layer."""

    *batch_dims, seq_length, hidden_size = layer_input.shape

    queries = hk.Linear(
        self.hidden_size,
        w_init=hk.initializers.TruncatedNormal(self.initializer_range),
        name='query_%d' % layer_index)(
            layer_input)
    keys = hk.Linear(
        self.hidden_size,
        w_init=hk.initializers.TruncatedNormal(self.initializer_range),
        name='keys_%d' % layer_index)(
            layer_input)
    values = hk.Linear(
        self.hidden_size,
        w_init=hk.initializers.TruncatedNormal(self.initializer_range),
        name='values_%d' % layer_index)(
            layer_input)

    btnh = (*batch_dims, seq_length, self.num_attention_heads,
            self.size_per_head)
    queries = jnp.reshape(queries, btnh)
    keys = jnp.reshape(keys, btnh)
    values = jnp.reshape(values, btnh)

    # Attention scores.
    attention_scores = jnp.einsum('...tnh,...fnh->...nft', keys, queries)
    attention_scores *= self.size_per_head**(-0.5)

    # attention_scores shape: [..., num_heads, num_attending, num_attended_over]
    # Broadcast the input mask along heads and query dimension.
    # If a key/value location is pad, do not attend over it.
    # Do that by plunging the attention logit to negative infinity.
    bcast_shape = list(input_mask.shape[:-1]) + [1, 1, input_mask.shape[-1]]
    input_mask_broadcasted = jnp.reshape(input_mask, bcast_shape)
    attention_mask = -1. * 1e30 * (1.0 - input_mask_broadcasted)
    attention_scores += attention_mask

    attention_probs = jax.nn.softmax(attention_scores)
    if is_training:
      attention_probs = hk.dropout(hk.next_rng_key(),
                                   self.attention_probs_dropout_prob,
                                   attention_probs)

    # Weighted sum.
    attention_output = jnp.einsum('...nft,...tnh->...fnh', attention_probs,
                                  values)
    attention_output = jnp.reshape(
        attention_output, (*batch_dims, seq_length, hidden_size))

    # Projection to hidden size.
    attention_output = hk.Linear(
        self.hidden_size,
        w_init=hk.initializers.TruncatedNormal(self.initializer_range),
        name='attention_output_dense_%d' % layer_index)(
            attention_output)
    if is_training:
      attention_output = hk.dropout(hk.next_rng_key(), self.hidden_dropout_prob,
                                    attention_output)
    attention_output = hk.LayerNorm(
        axis=-1,
        create_scale=True,
        create_offset=True,
        eps=TF_LAYERNORM_EPSILON,
        name='attention_output_ln_%d' % layer_index)(
            attention_output + layer_input)

    # FFW.
    intermediate_output = hk.Linear(
        self.intermediate_size,
        w_init=hk.initializers.TruncatedNormal(self.initializer_range),
        name='intermediate_output_%d' % layer_index)(
            attention_output)
    intermediate_output = jax.nn.gelu(intermediate_output)

    layer_output = hk.Linear(
        self.hidden_size,
        w_init=hk.initializers.TruncatedNormal(self.initializer_range),
        name='layer_output_%d' % layer_index)(
            intermediate_output)
    if is_training:
      layer_output = hk.dropout(hk.next_rng_key(), self.hidden_dropout_prob,
                                layer_output)
    layer_output = hk.LayerNorm(
        axis=-1,
        create_scale=True,
        create_offset=True,
        eps=TF_LAYERNORM_EPSILON,
        name='layer_output_ln_%d' % layer_index)(
            layer_output + attention_output)

    return layer_output

  def __call__(
      self,
      input_ids: jnp.DeviceArray,
      token_type_ids: tp.Optional[jnp.DeviceArray] = None,
      input_mask: tp.Optional[jnp.DeviceArray] = None,
      is_training: bool = True,
  ) -> networks.OutputWithPrior:
    """Forward pass of the BERT model."""

    # Prepare size, fill out missing inputs.
    *_, seq_length = input_ids.shape

    if input_mask is None:
      input_mask = jnp.ones(shape=input_ids.shape, dtype=jnp.int32)

    if token_type_ids is None:
      token_type_ids = jnp.zeros(shape=input_ids.shape, dtype=jnp.int32)

    position_ids = jnp.arange(seq_length)[None, :]

    # Embeddings.
    word_embedder = hk.Embed(
        vocab_size=self.vocab_size,
        embed_dim=self.hidden_size,
        w_init=hk.initializers.TruncatedNormal(self.initializer_range),
        name='word_embeddings')
    word_embeddings = word_embedder(input_ids)
    token_type_embeddings = hk.Embed(
        vocab_size=self.type_vocab_size,
        embed_dim=self.hidden_size,
        w_init=hk.initializers.TruncatedNormal(self.initializer_range),
        name='token_type_embeddings')(
            token_type_ids)
    position_embeddings = hk.Embed(
        vocab_size=self.max_position_embeddings,
        embed_dim=self.hidden_size,
        w_init=hk.initializers.TruncatedNormal(self.initializer_range),
        name='position_embeddings')(
            position_ids)
    input_embeddings = (
        word_embeddings + token_type_embeddings + position_embeddings)
    input_embeddings = hk.LayerNorm(
        axis=-1,
        create_scale=True,
        create_offset=True,
        eps=TF_LAYERNORM_EPSILON,
        name='embeddings_ln')(
            input_embeddings)
    if is_training:
      input_embeddings = hk.dropout(
          hk.next_rng_key(), self.hidden_dropout_prob, input_embeddings)

    # BERT layers.
    h = input_embeddings
    extra = {}
    for i in range(self.num_hidden_layers):
      h = self._bert_layer(
          h, layer_index=i, input_mask=input_mask, is_training=is_training)
      extra[f'hidden_layer_{i}'] = h
    last_layer = h

    # Masked language modelling logprobs.
    mlm_hidden = hk.Linear(
        self.hidden_size,
        w_init=hk.initializers.TruncatedNormal(self.initializer_range),
        name='mlm_dense')(last_layer)
    mlm_hidden = jax.nn.gelu(mlm_hidden)
    mlm_hidden = hk.LayerNorm(
        axis=-1,
        create_scale=True,
        create_offset=True,
        eps=TF_LAYERNORM_EPSILON,
        name='mlm_ln')(mlm_hidden)
    output_weights = jnp.transpose(word_embedder.embeddings)
    logits = jnp.matmul(mlm_hidden, output_weights)
    logits = hk.Bias(bias_dims=[-1], name='mlm_bias')(logits)
    log_probs = jax.nn.log_softmax(logits, axis=-1)

    # Pooled output: [CLS] token.
    first_token_last_layer = last_layer[..., 0, :]
    pooled_output = hk.Linear(
        self.hidden_size,
        w_init=hk.initializers.TruncatedNormal(self.initializer_range),
        name='pooler_dense')(
            first_token_last_layer)
    pooled_output = jnp.tanh(pooled_output)

    extra['logits'] = logits
    extra['log_probs'] = log_probs
    extra['pooled_output'] = pooled_output

    return networks.OutputWithPrior(
        train=pooled_output, prior=jnp.zeros_like(pooled_output), extra=extra)
