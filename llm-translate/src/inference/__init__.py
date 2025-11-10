"""
推理模块
"""

from .translator import Translator
from .beam_search import BeamSearch

__all__ = [
    'Translator',
    'BeamSearch'
]
