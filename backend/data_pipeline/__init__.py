"""
Data pipeline for building formality-labeled parallel corpora.

Sources: Korean OpenSubtitles, Tatoeba EN-KR
Pipeline: KoNLPy morphological analysis → EF suffix labeling → LLM augmentation
Output: TSV with (en, ko, formality) columns
"""
