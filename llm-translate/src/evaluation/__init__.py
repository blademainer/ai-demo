"""
评估模块
"""

from .metrics import compute_bleu, compute_metrics
from .evaluator import Evaluator

__all__ = [
    'compute_bleu',
    'compute_metrics',
    'Evaluator'
]
