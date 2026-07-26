import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
import math
import numpy as np


# Option 1: Linear Warmup + Cosine Decay (Most Common)
class WarmupCosineScheduler(LRScheduler):
    """Linear warmup then cosine annealing decay"""

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.0):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer)

    def get_lr(self):
        if self._step_count < self.warmup_steps:
            # Linear warmup
            alpha = self._step_count / self.warmup_steps
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = (self._step_count - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            progress = min(1.0, progress)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            min_lr = self.min_lr_ratio * self.base_lrs[0]
            return [
                min_lr + (base_lr - min_lr) * cosine_decay for base_lr in self.base_lrs
            ]


# Option 2: Linear Warmup + Linear Decay (Simple)
class WarmupLinearScheduler(LRScheduler):
    """Linear warmup then linear decay"""

    def __init__(self, optimizer, warmup_steps, total_steps):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer)

    def get_lr(self):
        if self._step_count < self.warmup_steps:
            # Linear warmup
            alpha = self._step_count / self.warmup_steps
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Linear decay
            progress = (self._step_count - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            progress = min(1.0, progress)
            return [base_lr * (1 - progress) for base_lr in self.base_lrs]


# Option 3: Exponential Warmup (More Conservative)
class WarmupExponentialScheduler(LRScheduler):
    """Exponential warmup then constant"""

    def __init__(self, optimizer, warmup_steps, total_steps, decay_rate=0.5):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.decay_rate = decay_rate
        super().__init__(optimizer)

    def get_lr(self):
        if self._step_count < self.warmup_steps:
            # Exponential warmup: 0.1 * base_lr to 1.0 * base_lr
            alpha = 0.1 + 0.9 * (
                1 - math.exp(-self._step_count / (self.warmup_steps * 0.3))
            )
            return [base_lr * min(1.0, alpha) for base_lr in self.base_lrs]
        else:
            # Gradual decay
            progress = (self._step_count - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            progress = min(1.0, progress)
            return [
                base_lr * (0.5 ** (progress / self.decay_rate))
                for base_lr in self.base_lrs
            ]
