__version__ = "0.0.1"
from pytorch_pretrained_bert.tokenization_gpt2 import GPT2Tokenizer
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, cached_path
from pytorch_pretrained_bert.modeling_gpt2 import GPT2Config, GPT2Model, GPT2Config
from pytorch_pretrained_bert.tokenization_gpt2 import GPT2Tokenizer

from .modeling_gpt2 import GPT2LMHeadModel
from .optim import Adam
from .loss import NLLLoss
from .trainer import Trainer
from .dataset import LineByLineTextDataset, TextDataset
from .metrics import AccuracyMetric, LossMetric, TokenAccuracyMetric
from .utils import get_logger, set_seed, get_args, get_dataset, get_optimizer
from .metrics import Metric
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm, trange
import numpy as np
import os
import sys
import json
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional


class GPT2(nn.Module):

    def __init__(self, config: GPT2Config):
        super().__init__()

        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([Block(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])

    def forward(self, input_ids=None, position_ids=None, token_type_ids=None, lm_labels=None):

        # The output weights are the same as the input embeddings, but there is an output-only bias for each token position embedding.#  # noqa: E501

        hidden_states = ()

        past = None

        if input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])

            position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None

            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None

            inputs_embeds = self.wte(input_ids)
            position_embeds = self.wpe(position_ids) if position_ids is not None else 0
            token_type_embeds = self.wte(token_type_ids) if token_type_ids is not None else 0

            embeds = inputs + position + token # noqa: E501

        hidden = self.drop(embeds)
        for i in range(self.config.n):
            hidden = self.h[i](hidden, past=past)
        output = hidden[:, -1, :] # [batch size, hidden size]

        return output