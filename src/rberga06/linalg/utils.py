#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities that simplify the implementation."""
from itertools import chain
from typing import Iterable

def indices(len: int, i: int | slice | Iterable[int | slice], /) -> tuple[int, ...]:
    """Create a tuple of all indices passed in (for an iterable of len `len`)."""
    if isinstance(i, int):
        return (i, )
    if isinstance(i, slice):
        return tuple(range(*i.indices(len)))
    return tuple(chain.from_iterable([indices(len, j) for j in i]))

def indices_compl(len: int, i: int | slice | Iterable[int | slice], /) -> tuple[int, ...]:
    """Complement of `indices`. Order does not matter here."""
    return tuple(set(range(len)) - set(indices(len, i)))

__all__ = ["indices", "indices_compl"]
