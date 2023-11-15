#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Concrete vectors (we all know and love)."""
from typing import Any, Iterable, Iterator, Literal, Self, cast, final, overload, override

from .utils import indices
from .abstract import C, R, AVec


@final
class Vec[K: (C, R)](AVec[K]):
    """A K^n vector (we all know and love)."""
    __slots__ = ("__v", "orient")

    # --- Attributes & Properties ---

    __v: list[K]
    orient: Literal["h", "v"]

    @property
    def n(self, /) -> int:
        """Vector length."""
        return len(self.__v)

    # --- Constructors ---

    def __init__(self, v: Iterable[K] = (), /, *, orient: Literal["h", "v"] = "h") -> None:
        self.__v = [*v]
        self.orient = orient

    @classmethod
    def fill(cls, k: K, len: int, /) -> Self:
        return cls([k]*len)

    @classmethod
    def null(cls, len: int, /) -> Self:
        return cls.fill(0, len)

    def copy(self, /) -> Self:
        return type(self)(self.__v, orient=self.orient)

    @property
    def T(self, /) -> Self:
        return type(self)(self.__v, orient="h" if self.orient == "v" else "v")

    def as_complex(self: "Vec[R]", /) -> "Vec[C]":
        """Interpret this (real) vector as a complex vector."""
        return self  # type: ignore

    # --- AVec implementation ---

    @override
    def __eq__(self, v: Any) -> bool:
        if not isinstance(v, Vec):
            return False
        return self.__v == cast(Vec[K], v).__v

    @override
    def __sum__(self, v: Self) -> Self:
        return type(self)(map(sum, zip(self.__v, v.__v)), orient=self.orient)

    @override
    def __mul__(self, k: K) -> Self:
        return type(self)((x*k for x in self.__v), orient=self.orient)

    # --- Concatenation ---

    def __or__(self, v: K | Self) -> Self:
        """Concatenate `v` to the right."""
        if isinstance(v, Vec):
            return type(self)((*self.__v, *v), orient=self.orient)
        if isinstance(v, complex | float | int):  # type: ignore[reportUnnecessaryIsinstance]
            return type(self)((*self.__v, v), orient=self.orient)
        return NotImplemented

    def __ror__(self, v: K, /) -> Self:
        """Concatenate `v` to the left."""
        if isinstance(v, complex | float | int):  # type: ignore[reportUnnecessaryIsinstance]
            return type(self)((v, *self.__v), orient=self.orient)
        return NotImplemented

    # --- Indexing ---

    @overload
    def __getitem__(self, k: int, /) -> K: ...
    @overload
    def __getitem__(self, k: slice | Iterable[int | slice], /) -> Self: ...
    def __getitem__(self, k: int | slice | Iterable[int | slice], /) -> K | Self:
        if isinstance(k, int):
            return self.__v[k]
        return type(self)((self.__v[i] for i in indices(self.n, k)), orient=self.orient)

    @overload
    def __setitem__(self, k: int, v: K, /) -> None: ...
    @overload
    def __setitem__(self, k: slice | Iterable[int | slice], v: Iterable[K], /) -> None: ...
    def __setitem__(self, k: int | slice | Iterable[int | slice], v: K | Iterable[K], /) -> None:
        if isinstance(k, int):
            self.__v[k] = cast(K, v)
        else:
            if v is self:
                # Copy the vector to avoid setting items before getting them
                v = self.copy()
            for i, x in zip(indices(self.n, k), cast(Iterable[K], v)):
                self.__v[i] = x

    # --- Other special methods ---

    def __iter__(self, /) -> Iterator[K]:
        return iter(self.__v)

    def __bool__(self, /) -> bool:
        """self != Vec.null(self.n)"""
        return self.__v != [0]*self.n

    def __len__(self, /) -> int:
        return self.n

    @override
    def __repr__(self, /) -> str:
        return "[" + ", ".join(map(repr, self.__v)) + "]"



def vec[K: (R, C)](v: Vec[K] | Iterable[K] | K, /) -> Vec[K]:
    """Convert the argument into a `Vec[K]`, unless it's already a `Vec[K]`."""
    if isinstance(v, Vec):
        return v
    if isinstance(v, Iterable):
        return Vec(v)
    return Vec((v,))



__all__ = ["Vec", "vec"]
