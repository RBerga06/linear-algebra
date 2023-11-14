#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""(Abstract) Vector Spaces."""
from abc import abstractmethod
from typing import Any, Protocol, Self, override

type R = float    # real field
type C = complex  # complex field


class AVec[K: (C, R)](Protocol):
    """Abstract Vector."""

    @abstractmethod
    @override
    def __eq__(self, v: Any, /) -> bool: ...
    @abstractmethod
    def __sum__(self, v: Self, /) -> Self: ...
    @abstractmethod
    def __mul__(self, k: K, /) -> Self: ...

    def __rsum__(self, v: Self) -> Self:
        return self * v  # type: ignore

    def __rmul__(self, k: K) -> Self:
        return self * k  # type: ignore

    def __pos__(self, /) -> Self:
        """+ self"""
        return self

    def __neg__(self, /) -> Self:
        """- self"""
        return self * -1  # type: ignore


__all__ = [
    "R", "C",
    "AVec",
]
