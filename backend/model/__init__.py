"""
mBART-50 fine-tuning for formality-conditioned translation.

Expands the tokenizer with <formal>, <polite>, <casual> conditioning tokens,
resizes embedding layers, and trains on formality-labeled parallel data.
"""
