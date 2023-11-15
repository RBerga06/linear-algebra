#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Matrices!"""
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Any, Iterable, Iterator, Self, cast, final, overload, override

from .utils import indices, indices_compl
from .abstract import C, R, AVec
from .vector import Vec


@final
class Mat[K: (C, R)](AVec[K]):
    """A Mmxn(K) matrix (we all know and love)."""
    __slots__ = ("__m", )

    # --- Attributes & Properties ---

    __m: list[list[K]]

    @property
    def m(self, /) -> int:
        """Vector length."""
        return len(self.__m)

    @property
    def n(self, /) -> int:
        """Vector length."""
        if self.__m:
            return len(self.__m[0])
        return 0

    # --- Constructors ---

    def __init__(self, m: Iterable[Iterable[K]] = (), /) -> None:
        self.__m = [[*r] for r in m]

    @classmethod
    def from_rows(cls, rows: Iterable[Iterable[K]], /) -> Self:
        return cls(rows)

    @classmethod
    def from_cols(cls, cols: Iterable[Iterable[K]], /) -> Self:
        return cls(zip(*cols))

    @classmethod
    def fill(cls, k: K, m: int, n: int | None = None, /) -> Self:
        return cls([[k]*(m if n is None else n)]*m)

    @classmethod
    def null(cls, m: int, n: int | None = None, /) -> Self:
        return cls.fill(0, m, n)

    @classmethod
    def Id(cls, n: int, /) -> Self:
        return cls([[0]*i+[1]+[0]*(n-i-1) for i in range(n)])  # type: ignore

    def copy(self, /) -> Self:
        return type(self)(self.__m)

    @property
    def T(self, /) -> Self:
        return type(self)(self.__m)

    def as_complex(self: "Mat[R]", /) -> "Mat[C]":
        """Interpret this (real) vector as a complex vector."""
        return self  # type: ignore

    # --- Misc mutating methods ---

    def swap_rows(self, r1: int, r2: int, /) -> None:
        """Swap two rows."""
        self.__m[r1], self.__m[r2] = self.__m[r2], self.__m[r1]

    def swap_cols(self, c1: int, c2: int, /) -> None:
        """Swap two columns."""
        # TODO: Re-implement this in a more efficient way
        self[:,(c1,c2)] = self[:,(c2,c1)]

    # --- Iterators ----

    def rows(self, /) -> list[Vec[K]]:
        return [Vec(row, orient="h") for row in self.__m]

    def cols(self, /) -> list[Vec[K]]:
        return [Vec(col, orient="v") for col in zip(*self.__m)]

    # --- AVec implementation ---

    @override
    def __eq__(self, m: Any) -> bool:
        if not isinstance(m, Mat):
            return False
        return self.__m == cast(Mat[K], m).__m

    @override
    def __sum__(self, m: Self, /) -> Self:
        return type(self)([[x1+x2 for x1, x2 in zip(r1, r2)] for r1, r2 in zip(self.__m, m.__m)])

    @override
    def __mul__(self, k: K, /) -> Self:
        return type(self)([[x*k for x in r] for r in self.__m])

    # --- Matrix multiplication ---

    def __matmul__(self, m: Self, /) -> Self:
        return type(self)([
            [sum([self.__m[i][k] * m.__m[k][j] for k in range(self.n)]) for j in range(m.n)] for i in range(self.m)
        ])

    # --- Concatenation ---

    def __or__(self, m: Self | Vec[K], /) -> Self:
        """Concatenate `m` to the right."""
        return type(self)(zip(  # Mat.from_cols(
            *zip(*self.__m),    #  *self.cols(),
            *zip(*mat(m).__m),  #  *mat(m).cols(),
        ))                      # )

    def __ror__(self, m: Self | Vec[K], /) -> Self:
        """Concatenate `m` to the left."""
        return type(self)(zip(  # Mat.from_cols(
            *zip(*mat(m).__m),  #  *mat(m).cols(),
            *zip(*self.__m),    #  *self.cols(),
        ))                      # )

    def __and__(self, x: Self | Vec[K], /) -> Self:
        """Concatenate `m` below."""
        return type(self)([*self.__m, *mat(x).__m])

    def __rand__(self, x: Self | Vec[K], /) -> Self:
        """Concatenate `m` above."""
        return type(self)([*mat(x).__m, *self.__m])

    # --- Indexing ---

    @overload
    def __getitem__(self, key: tuple[int, int], /) -> K: ...
    @overload
    def __getitem__(self, key: tuple[int, slice | Iterable[int | slice]], /) -> Vec[K]: ...
    @overload
    def __getitem__(self, key: tuple[slice | Iterable[int | slice], int], /) -> Vec[K]: ...
    @overload
    def __getitem__(self, key: tuple[slice | Iterable[int | slice], slice | Iterable[int | slice]], /) -> Self: ...
    def __getitem__(self, key: tuple[int | slice | Iterable[int | slice], int | slice | Iterable[int | slice]], /) -> Self | Vec[K] | K:
        ii, jj = key
        if isinstance(ii, int):
            if isinstance(jj, int):
                return self.__m[ii][jj]
            return Vec([self.__m[ii][j] for j in indices(self.n, jj)], orient="h")
        if isinstance(jj, int):
            return Vec([self.__m[i][jj] for i in indices(self.m, ii)], orient="v")
        return type(self)([[self.__m[i][j] for j in indices(self.n, jj)] for i in indices(self.m, ii)])

    @overload
    def __setitem__(self, key: tuple[int, int], val: K, /) -> None: ...
    @overload
    def __setitem__(self, key: tuple[int, slice | Iterable[int | slice]], val: Iterable[K], /) -> None: ...
    @overload
    def __setitem__(self, key: tuple[slice | Iterable[int | slice], int], val: Iterable[K], /) -> None: ...
    @overload
    def __setitem__(self, key: tuple[slice | Iterable[int | slice], slice | Iterable[int | slice]], val: Iterable[Iterable[K]], /) -> None: ...
    def __setitem__(self, key: tuple[int | slice | Iterable[int | slice], int | slice | Iterable[int | slice]], val: Iterable[Iterable[K]] | Iterable[K] | K, /) -> None:
        ii, jj = key
        if isinstance(ii, int):
            if isinstance(jj, int):
                self.__m[ii][jj] = cast(K, val)
            else:
                for j, x in zip(indices(self.n, jj), cast(Iterable[K], val)):
                    self.__m[ii][j] = x
        else:
            if isinstance(jj, int):
                for i, x in zip(indices(self.m, ii), cast(Iterable[K], val)):
                    self.__m[i][jj] = x
            if val is self:
                # Make sure we don't mutate `val` while iterating on it
                val = self.copy()
            for i, row in zip(indices(self.m, ii), cast(Iterable[Iterable[K]], val)):
                for j, x in zip(indices(self.n, jj), row):
                    self.__m[i][j] = x

    @property
    def without(self, /) -> "_MatWithoutAccessHelper[K]":
        return _MatWithoutAccessHelper(self)

    # --- Other special methods ---

    def __iter__(self, /) -> Iterator[Vec[K]]:
        return iter(self.rows())

    def __bool__(self, /) -> bool:
        """self != Matrix.null(self.m, self.n)"""
        return self.__m != [[0]*self.m]*self.n

    @override
    def __repr__(self, /) -> str:
        if self.m == 0:
            return "[]"
        if self.m == 1:
            return "[" + "\t".join(map(repr, self.__m[0])) + "\t]"
        return "\n".join([
            "⎡" + "\t".join(map(repr, self.__m[ 0])) + "\t⎤", *[
            "⎢" + "\t".join(map(repr, self.__m[ i])) + "\t⎢" for i in range(1, self.m - 1)],
            "⎣" + "\t".join(map(repr, self.__m[-1])) + "\t⎦",
        ])

    # --- Matrix-specific operations ---

    def gauss(self, /) -> "Mat[K]":
        """Run the Gauss algorithm (always returns a new matrix)."""
        # Se la matrice è nulla, abbiamo finito
        if not self:
            return self.copy()
        # Se la prima colonna è nulla, saltala
        if not self[:,0]:
            return self[:,0] | self[:,1:].gauss()
        # Fintanto che il primo elemento è uno 0, scambia la prima riga con un'altra
        i = 1
        while not self[0,0]:
            self[(0,i),:] = self[(i,0),:]  # scambia le righe 0 e i
            i += 1
        # Ora il primo elemento è sicuramente un perno. Sottraiamo le volte necessarie ogni riga
        r = self[0,:]/self[0,0]
        for i in range(1, self.m):
            self[i,:] -= r * self[i,0]
        # Ora la prima colonna è tutta di zeri (a parte il perno): procedi senza prima riga e prima colonna
        return self[0,:] & (self[1:,0] | self[1:,1:].gauss())

    def det(self, /) -> K:
        """Evaluate the determinant."""
        if self.n != self.m:
            raise ValueError("The determinant is not defined for a non-square matrix.")
        m = self.gauss()
        return reduce(mul, [m.__m[i][i] for i in range(self.n)])

    @property
    def invertible(self, /) -> bool:
        if self.n != self.m:  # non-square matrices cannot be invertible
            return False
        return self.det() != 0

    @property
    def inverse(self, /) -> "Mat[K]":
        """The inverse matrix (if self is invertible)"""
        if not self.invertible:
            raise ValueError("Cannot find the inverse of a non-invertible matrix.")
        return Mat[K]([[(((i+j)%2)*2-1)*self.without[i,j].det()/self.det() for j in range(self.n)] for i in range(self.m)])


@final
@dataclass(slots=True, frozen=True)
class _MatWithoutAccessHelper[K: (C, R)]:
    mat: Mat[K]

    def __getitem__(self, key: tuple[int | slice | Iterable[int | slice], int | slice | Iterable[int | slice]]) -> Mat[K]:
        i, j = key
        return self.mat[indices_compl(self.mat.m, i), indices_compl(self.mat.n, j)]

    def __setitem__(
        self,
        key: tuple[int | slice | Iterable[int | slice], int | slice | Iterable[int | slice]],
        val: Iterable[Iterable[K]],
    /) -> None:
        i, j = key
        self.mat[indices_compl(self.mat.m, i),indices_compl(self.mat.n, j)] = val


def mat[K: (R, C)](m: K | Vec[K] | Iterable[K] | Mat[K] | Iterable[Iterable[K]], /) -> Mat[K]:
    if isinstance(m, Mat):
        return cast(Mat[K], m)
    if isinstance(m, Vec):
        m = cast(Vec[K], m)
        match m.orient:
            case "h":
                return Mat.from_rows([m])
            case "v":
                return Mat.from_cols([m])
    if isinstance(m, int | float | complex):
        return Mat(((m,),))
    it = [*m]
    if (not it) or isinstance(it[0], Iterable):
        return Mat[K](it)  # type: ignore
    return Mat.from_cols([it])  # type: ignore


__all__ = ["Mat", "mat"]
