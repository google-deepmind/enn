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
"""Utilities for logging to the terminal.

Forked from third_party/py/acme/utils/loggers/terminal.py
"""

import logging
import time
import typing as tp

import numpy as np
import tree
import typing_extensions


LoggingData = tp.Mapping[str, tp.Any]


class Logger(typing_extensions.Protocol):
  """A logger has a `write` method."""

  def write(self, data: LoggingData):
    """Writes `data` to destination (file, terminal, database, etc)."""


def _tensor_to_numpy(value: tp.Any):
  if hasattr(value, 'numpy'):
    return value.numpy()  # tf.Tensor (TF2).
  if hasattr(value, 'device_buffer'):
    return np.asarray(value)  # jnp.DeviceArray.
  return value


def _to_numpy(values: tp.Any):
  """Converts tensors in a nested structure to numpy.

  Converts tensors from TensorFlow to Numpy if needed without importing TF
  dependency.

  Args:
    values: nested structure with numpy and / or TF tensors.

  Returns:
    Same nested structure as values, but with numpy tensors.
  """
  return tree.map_structure(_tensor_to_numpy, values)


def _format_key(key: str) -> str:
  """Internal function for formatting keys."""
  return key.replace('_', ' ').title()


def _format_value(value: tp.Any) -> str:
  """Internal function for formatting values."""
  value = _to_numpy(value)
  if isinstance(value, (float, np.number)):
    return f'{value:0.3f}'
  return f'{value}'


def serialize(values: LoggingData) -> str:
  """Converts `values` to a pretty-printed string.

  This takes a dictionary `values` whose keys are strings and returns
  a formatted string such that each [key, value] pair is separated by ' = ' and
  each entry is separated by ' | '. The keys are sorted alphabetically to ensure
  a consistent order, and snake case is split into words.

  For example:

      values = {'a': 1, 'b' = 2.33333333, 'c': 'hello', 'big_value': 10}
      # Returns 'A = 1 | B = 2.333 | Big Value = 10 | C = hello'
      values_string = serialize(values)

  Args:
    values: A dictionary with string keys.

  Returns:
    A formatted string.
  """
  return ' | '.join(f'{_format_key(k)} = {_format_value(v)}'
                    for k, v in sorted(values.items()))


class TerminalLogger:
  """Logs to terminal."""

  def __init__(
      self,
      label: str = '',
      print_fn: tp.Callable[[str], None] = logging.info,
      serialize_fn: tp.Callable[[LoggingData], str] = serialize,
      time_delta: float = 0.0,
  ):
    """Initializes the logger.

    Args:
      label: label string to use when logging.
      print_fn: function to call which acts like print.
      serialize_fn: function to call which transforms values into a str.
      time_delta: How often (in seconds) to write values. This can be used to
        minimize terminal spam, but is 0 by default---ie everything is written.
    """

    self._print_fn = print_fn
    self._serialize_fn = serialize_fn
    self._label = label and f'[{_format_key(label)}] '
    self._time = time.time()
    self._time_delta = time_delta

  def write(self, data: LoggingData):
    now = time.time()
    if (now - self._time) > self._time_delta:
      self._print_fn(f'{self._label}{self._serialize_fn(data)}')
      self._time = now


def make_default_logger(label: str, time_delta: float = 0.0) -> TerminalLogger:
  return TerminalLogger(label=label, time_delta=time_delta)
