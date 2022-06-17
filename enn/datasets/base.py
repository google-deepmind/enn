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
"""Interface for datasets in jaxline."""

import abc
import dataclasses
import enum
from typing import Callable, Dict, Generator, List, Sequence, Union
from enn import base
import numpy as np
import tensorflow.compat.v2 as tf

Array = Union[np.ndarray, tf.Tensor]
# DatasetDict is a Dict with "images" and "labels" as keys
DatasetDict = Dict[str, Array]

# TODO(author3): Describe DatasetGenerator
DatasetGenerator = Generator[base.Batch, None, None]
DatasetTransformer = Callable[[tf.data.Dataset], tf.data.Dataset]


class Dataset(abc.ABC):
  """Abstract base class of a dataset."""

  @property
  @abc.abstractmethod
  def num_classes(self) -> int:
    """Number of output classes."""

  @property
  @abc.abstractmethod
  def eval_input_shape(self) -> Sequence[int]:
    """Returns the shape of a single eval input from the dataset."""

  @abc.abstractmethod
  def train_dataset(self) -> DatasetGenerator:
    """Returns the train dataset."""

  @abc.abstractmethod
  def eval_datasets(self) -> Dict[str, DatasetGenerator]:
    """Returns a dictionary of eval datasets.

    The keys for these datasets should correspond to the self.mode in jaxline.
    """


@dataclasses.dataclass
class DatasetWithTransform(Dataset):
  """Dataset that implements dataset transforms explicitly on training/eval.

  The point of this class is to allow for explicit *interception* of batches
  so that we can more easily implement OOD experiments.
  """
  train_ds_transformer: DatasetTransformer
  eval_ds_transformers: Dict[str, DatasetTransformer]


class OodVariant(enum.Enum):
  WHOLE: str = 'eval'
  IN_DISTRIBUTION: str = 'eval_in_dist'
  OUT_DISTRIBUTION: str = 'eval_out_dist'

  @classmethod
  def valid_values(cls) -> List[str]:
    return list(map(lambda c: c.value, cls))


EVAL_TRANSFORMERS_DEFAULT = dataclasses.field(
    default_factory=lambda: {OodVariant.WHOLE.value: (lambda x: x)})
