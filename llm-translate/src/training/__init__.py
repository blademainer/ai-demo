"""
训练模块
"""

from .trainer import Trainer
from .scheduler import NoamLR
from .loss import LabelSmoothingCrossEntropy

__all__ = [
    'Trainer',
    'NoamLR',
    'LabelSmoothingCrossEntropy'
]
