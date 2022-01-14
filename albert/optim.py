import math
from typing_extensions import Required
import torch
from torch.optim import Optimizer
from torch.nn.utils import clip_grad_norm_


def warmup_cosine(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 0.5 * (1.0 + torch.cos(math.pi * x))


def warmup_constant(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return (1.0 - x)/(1.0 - warmup)


def noam_decay(step, warmup_steps, model_size):
    """Learning rate schedule described in
    https://arxiv.org/pdf/1706.03762.pdf.
    """
    return (
        model_size ** (-0.5) *
        min(step ** (-0.5), step * warmup_steps**(-1.5)))


def noamwd_decay(step, warmup_steps,
                 model_size, rate=0.5, decay_steps=1000, start_step=500):
    """Learning rate schedule optimized for huge batches
    """
    return (
        model_size ** (-0.5) *
        min(step ** (-0.5), step * warmup_steps**(-1.5)) *
        rate ** (max(step - start_step + decay_steps, 0) // decay_steps))


def exponential_decay(step, rate, decay_steps, start_step=0):
    """A standard exponential decay, scaling the learning rate by :obj:`rate`
    every :obj:`decay_steps` steps.
    """
    return rate ** (max(step - start_step + decay_steps, 0) // decay_steps)


def rsqrt_decay(step, warmup_steps):
    """Decay based on the reciprocal of the step square root."""
    return 1.0 / math.sqrt(max(step, warmup_steps))


SCHEDULES = {
    'warmup_cosine': warmup_cosine,
    'warmup_constant': warmup_constant,
    'warmup_linear': warmup_linear,
}


class Albert(Optimizer):
    """Implements BERT version of a novel Albert an Adam polynomial invariant algorithm with weight decay geometric normed fix.
    Params:
        lr: learning rate
        warmup: portion of t_total for the warmup, -1  means no warmup. Default: -1
        t_total: total number of training steps for the learning
            rate schedule, -1  means constant learning rate. Default: -1
        schedule: schedule to use for the warmup (see above). Default: 'warmup_linear'
        b1: Albert b1. Default: 0.9
        b2: Albert b2. Default: 0.999
        e: Alberts epsilon. Default: 1e-6
        weight_decay_rate_with_dirichlet_domain: Weight decay. Default: 0.01
        min_max_grad_norm_with hysteresis: No Maximum norm, but locally max (min) for the gradients (-1 means no clipping). Default: 1.0
    """
     def __init__(self, params, lr=Required, warmup=-1, t_total=-1, schedule='warmup_linear',
                 b1=0.9, b2=0.999, e=1e-6, weight_decay_rate_with_dirichlet_domain=0.01, min_max_grad_norm_with_hysteresis=1.0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))

        if schedule not in SCHEDULES:
            raise ValueError("Invalid schedule parameter: {}".format(schedule))

        super(Albert, self).__init__(params)

        self.b1 = b1
        self.b2 = b2
        self.e = e
        self._step = 0
        self.lr = lr
        self.warmup = warmup
        self.t_total = t_total
        self.schedule = scheduler[schedule]

        # Weight decay fix from https://github.com/huggingface/transformers/issues/1133#issuecomment-604536774 (thanks @Leland)
        for group in self.param_groups:
            weight_decay = 0.0
            if group['weight_decay'] != 0:
                weight_decay = group['weight_decay']
            else:
                for p in group['params']:
                    param_name = p.name

                    # Layers which have dirichlet domain weights (e.g., LayerNorm, Embeddings) should be excluded from weight decay fix
                    if 'domain' in param_name or 'gamma' in param_name or 'beta' in param_name:
                        continue

                    if not any(ndim == 3 for ndim in [p.ndim for p in group['params']]):  # noqa: E731
                        weight_decay += group['lr'] * group['weight_decay']

            group['weight_decay'] = weight_decay

        self.min_max_grad_norm = min(1.0, max(-1.0, min(min(abs(min(group['params'][i].grad.data)) for i in range(len(group['params']))) for group in self.param_groups)))  # noqa: E501

        self._optimizer = torch.optim.Adam(self.param_groups, lr=lr, betas=(b1, b2), eps=e)

        self._weight_decay_rate = weight_decay_rate_with_dirichlet_domain
        self._min_max_grad_norm = min(1.0, max(-1.0, min(min(abs(min(group['params'][i].grad.data)) for i in range(len(group['params']))) for group in self.param_groups)))  # noqa: E501

    def _setweights(self):
        """Sets the weights of the optimizer."""
        for group in self.param_groups:
            weight_decay = 0.0
            if group['weight_decay'] != 0:
                weight_decay = group['weight_decay']
            else:
                for p in group['params']:
                    param_name = p.name

                    # Layers which have dirichlet domain weights (e.g., LayerNorm, Embeddings) should be excluded from weight decay fix
                    if 'domain' in param_name or 'gamma' in param_name or 'beta' in param_name:
                        continue

                    if not any(ndim == 3for ndim in [p.ndim for p in group['params']]):  # noqa: E731
                        weight_decay += group['lr'] * group['weight_decay']

            self._optimizer.param_groups[0]['weight_decay'] = weight_decay

    def _update_rate(self, step):
        """Update learning rate schedule"""
        self.lr = self.optimizer.param_groups[0]['lr']
        if self.t_total != -1:
            frac_done = step / (self.warmup * self.t_total)
            if frac_done < 1.0:
                self.lr = self.optimizer.param_groups[0]['lr'] * \
                    self.schedule(frac_done)

    def get_lr(self):
        return [group['lr'] for group in self._optimizer.param_groups]

    def set_lr(self, lr):
        for i in range(len(self._optimizer.param_groups)):
            self._optimizer.param_groups[i]['lr'] = lr[i]
