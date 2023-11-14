#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for `rberga06.linalg.vector`"""
from rberga06.linalg import R, Vec, vec

class TestVec:
    def test_new(self, /) -> None:
        assert vec(()) == Vec()
        assert not Vec()
        assert not Vec.null(5)

    def test_getitem(self, /) -> None:
        v = Vec(range(5))
        assert v[0] == 0
        assert v[1] == 1
        assert v[4] == 4
        assert v[:] == v
        assert v[2:] == Vec([2, 3, 4])
        assert v[:3] == Vec([0, 1, 2])
        assert v[::2] == Vec([0, 2, 4])
        assert v[1:-1] == Vec([1, 2, 3])
        assert v[0,1] == Vec([0, 1])
        assert v[1,0] == Vec([1, 0])
        assert v[0,2:4] == Vec([0, 2, 3])
        assert v[0,0,0] == Vec([0, 0, 0])

    def test_setitem(self, /) -> None:
        v = Vec[R].null(5)
        v[0] = 1
        assert v == Vec([1, 0, 0, 0, 0])
        v[:] = range(5)
        assert v == Vec([0, 1, 2, 3, 4])
        v[1:3,0,-1:-3:-1] = v[:]
        # equivalent to:
        #   tmp = v.copy()
        #   v[1] = tmp[0]
        #   v[2] = tmp[1]
        #   v[0] = tmp[2]
        #   v[4] = tmp[3]
        #   v[3] = tmp[4]
        assert v == Vec([2, 0, 1, 4, 3])
