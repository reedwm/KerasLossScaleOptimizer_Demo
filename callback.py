# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=g-import-not-at-top
"""Callbacks: utilities called at certain points during model training.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import csv
import io
import json
import os
import re
import tempfile
import time

import numpy as np
import six

from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.distribute import distribute_coordinator_context as dc_context
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.python.keras.utils.generic_utils import Progbar
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import checkpoint_management
from tensorflow.python.util.tf_export import keras_export

class ProgbarLogger(Callback):
    """Callback that prints metrics to stdout.
    # Arguments
        count_mode: One of "steps" or "samples".
            Whether the progress bar should
            count samples seen or steps (batches) seen.
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over an epoch.
            Metrics in this list will be logged as-is.
            All others will be averaged over time (e.g. loss, etc).
    # Raises
        ValueError: In case of invalid `count_mode`.
    """

    def __init__(self, count_mode='samples',
                 stateful_metrics=None):
        super(ProgbarLogger, self).__init__()
        if count_mode == 'samples':
            self.use_steps = False
        elif count_mode == 'steps':
            self.use_steps = True
        else:
            raise ValueError('Unknown `count_mode`: ' + str(count_mode))
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

    def on_train_batch_begin(self, batch, logs=None):
        self._batch_begin(batch, logs)

    def on_train_batch_end(self, batch, logs=None):
        self._batch_end(batch, logs)

    def on_test_batch_begin(self, batch, logs=None):
        self._batch_begin(batch, logs)

    def on_test_batch_end(self, batch, logs=None):
        self._batch_end(batch, logs)

    def _batch_begin(self, batch, logs):
        if self.seen < self.target:
            self.log_values = []

    def _batch_end(self, batch, logs):
        logs = logs or {}
        batch_size = logs.get('size', 0)
        if self.use_steps:
            self.seen += 1
        else:
            self.seen += batch_size

        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))

        # Skip progbar update for the last batch;
        # will be handled by on_epoch_end.
        self.progbar.update(self.seen, self.log_values)

    def on_train_begin(self, epoch, logs=None):
        if self.use_steps:
            target = self.params['steps']
        else:
            target = self.params['samples']

        self.target = target
        self.progbar = Progbar(target=self.target,
                               verbose=1,
                               stateful_metrics=self.stateful_metrics,
                               interval=5e-4)
        self.seen = 0

    def on_train_end(self, epoch, logs=None):
        logs = logs or {}
        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))

        self.progbar.update(self.seen, self.log_values)
        del self.progbar

    def on_test_begin(self, logs=None):
        if self.use_steps:
            target = self.params['steps']
        else:
            target = self.params['steps'] * self.params["batch_size"]

        print("TARGET:", target)
        print("PARAMS:", self.params)

        self.target = target
        self.progbar = Progbar(target=self.target,
                               verbose=1,
                               stateful_metrics=self.stateful_metrics,
                               interval=5e-4)
        self.seen = 0

    def on_test_end(self, logs=None):
        logs = logs or {}
        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))

        self.progbar.update(self.seen, self.log_values)
        del self.progbar
