"""hadamax — Hadamard + Lloyd-Max compressed vector index.

Fast approximate nearest-neighbor search via randomized Hadamard transform
and optimal Gaussian scalar quantization.  Pure NumPy, no heavy dependencies.

    >>> from hadamax import HadaMaxIndex
    >>> idx = HadaMaxIndex(dim=384, bits=4)
    >>> idx.add_batch(ids, vectors)
    >>> results = idx.search(query, k=10)

See https://github.com/stffns/hadamax for documentation.
"""
from __future__ import annotations

from ._index import HadaMaxIndex
from ._codebooks import get_codebook
from ._rotation import rht, padded_dim

__version__ = "0.1.0"
__all__ = ["HadaMaxIndex", "get_codebook", "rht", "padded_dim"]
