# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import torch.nn as nn
from ops import FactorizedReduce, ConvBranch, PoolBranch

_counter = 0

def global_mutable_counting():
    """
    A program level counter starting from 1.
    """
    global _counter
    _counter += 1
    return _counter

with open("macro.json") as f:
    fixed_arc = json.load(f)
class LayerChoice(nn.Module):
  def __init__(self, op_choices, key=None, **kwargs):
    super().__init__()
    key = self.__class__.__name__ + str(global_mutable_counting())
    self.ops = nn.ModuleList()
    assert len(fixed_arc[key]) == len(op_choices)
    for weight, op in zip(fixed_arc[key], op_choices):
      if weight:
        self.ops.append(op)
    assert self.ops
  def forward(self, x):
    return sum(op(x) for op in self.ops)
class InputChoice(nn.Module):
  def __init__(self, key=None, **kwargs):
    super().__init__()
    key = self.__class__.__name__ + str(global_mutable_counting())
    self.input_weight = fixed_arc[key]
  def forward(self, tensor_list):
    assert len(self.input_weight) == len(tensor_list)
    return sum(t for w, t in zip(self.input_weight, tensor_list) if w)

class ENASLayer(nn.Module):

    def __init__(self, key, prev_labels, in_filters, out_filters):
        super().__init__()
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.mutable = LayerChoice([
            ConvBranch(in_filters, out_filters, 3, 1, 1, separable=False),
            ConvBranch(in_filters, out_filters, 3, 1, 1, separable=True),
            ConvBranch(in_filters, out_filters, 5, 1, 2, separable=False),
            ConvBranch(in_filters, out_filters, 5, 1, 2, separable=True),
            PoolBranch('avg', in_filters, out_filters, 3, 1, 1),
            PoolBranch('max', in_filters, out_filters, 3, 1, 1)
        ])
        if len(prev_labels) > 0:
            self.skipconnect = InputChoice(choose_from=prev_labels, n_chosen=None)
        else:
            self.skipconnect = None
        self.batch_norm = nn.BatchNorm2d(out_filters, affine=False)

    def forward(self, prev_layers):
        out = self.mutable(prev_layers[-1])
        if self.skipconnect is not None:
            connection = self.skipconnect(prev_layers[:-1])
            if connection is not None:
                out += connection
        return self.batch_norm(out)


class GeneralNetwork(nn.Module):
    def __init__(self, num_layers=12, out_filters=24, in_channels=3, num_classes=10,
                 dropout_rate=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.out_filters = out_filters

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_filters, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_filters)
        )

        pool_distance = self.num_layers // 3
        self.pool_layers_idx = [pool_distance - 1, 2 * pool_distance - 1]
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)

        self.layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        labels = []
        for layer_id in range(self.num_layers):
            labels.append("layer_{}".format(layer_id))
            if layer_id in self.pool_layers_idx:
                self.pool_layers.append(FactorizedReduce(self.out_filters, self.out_filters))
            self.layers.append(ENASLayer(labels[-1], labels[:-1], self.out_filters, self.out_filters))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dense = nn.Linear(self.out_filters, self.num_classes)

    def forward(self, x):
        bs = x.size(0)
        cur = self.stem(x)

        layers = [cur]

        for layer_id in range(self.num_layers):
            cur = self.layers[layer_id](layers)
            layers.append(cur)
            if layer_id in self.pool_layers_idx:
                for i, layer in enumerate(layers):
                    layers[i] = self.pool_layers[self.pool_layers_idx.index(layer_id)](layer)
                cur = layers[-1]

        cur = self.gap(cur).view(bs, -1)
        cur = self.dropout(cur)
        logits = self.dense(cur)
        return logits
