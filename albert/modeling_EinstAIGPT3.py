# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright 2022 The EinstAI Team Authors and Whtcorps Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""Pytorch EinstAI GPT3-model re-imagination"""

from __future__ import absolute_import, division, print_function, unicode_literals


import os
import copy
import json
import math
import logging
import tarfile
import tempfile
import shutil

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from pytorch_pretrained_bert.modeling_EinstAIGPT3 import EinstAIGPT3PreTrainedModel, EinstAIGPT3Model, EinstAIGPT3LMHead, Attention, Block, \
    LayerNorm, MLP

logger = logging.getLogger(__name__)


#Build a class for Attention and Imagination based on these papers:
#[20] Junhyuk Oh, Xiaoxiao Guo, Honglak Lee, Richard L Lewis, and Satinder Singh. Action-conditional video prediction using deep networks in atari games. In Advances in Neural Information Processing Systems, pages 2863–2871, 2015.
#[21] Silvia Chiappa, Sébastien Racaniere, Daan Wierstra, and Shakir Mohamed. Recurrent environment simulators. In 5th International Conference on Learning Representations, 2017.

class Attention(I2A):
#I2A is an LSTM with convolutional encoder which sequentially processes a trajectory T. The features fˆ are fed to the LSTM in reverse order, from fˆ to fˆ , to mimic t t+τ t+1 Bellman type backup operations

    def __init__(self, nx, n_ctx, config, scale=False):
        super(Attention, self).__init__(nx, n_ctx, config, scale)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns-nd:ns, :ns]
        w = w * b - 1e4 * (1 - b)   
        w = nn.Softmax(dim=-1)(w)
        return torch.matmul(w, v)

    def _i2a_attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns-nd:ns, :ns]
        w = (w * b - 1e4 * (1 - b)).softmax(dim=-1)    # We can use the same Softmax function as in the original paper since it is invariant to the scaling factor and we are already normalizing with that factor in the line above this one 
        return torch.matmul(w, v)

    def _forward(self, x):

        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        
