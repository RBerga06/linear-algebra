#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""(Abstract) Vector Spaces."""
from abc import abstractmethod
from typing import Any, Protocol, Self, cast, override

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
        return self.__sum__(v)

    def __rmul__(self, k: K) -> Self:
        return self.__mul__(k)

    def __sub__(self, v: Self, /) -> Self:
        return self.__sum__(v.__neg__())

    def __truediv__(self, k: K, /) -> Self:
        return self.__mul__(1/k)

    def __pos__(self, /) -> Self:
        """+ self"""
        return self

    def __neg__(self, /) -> Self:
        """- self"""
        return self.__mul__(cast(K, -1))


__all__ = [
    "R", "C",
    "AVec",
]
