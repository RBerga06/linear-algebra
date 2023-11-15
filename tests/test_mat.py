#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for `rberga06.linalg.vector`"""
from rberga06.linalg import Vec, Mat, mat

class TestMat:
    def test_new(self, /) -> None:
        assert mat([[1]]) == mat([1]) == mat(Vec([1])) == mat(Mat([[1]])) == Mat([[1]])

    def test_getitem(self, /) -> None:
        m = Mat([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        for i in range(3):
            for j in range(3):
                assert m[i,j] == 3*i + j + 1
            assert m[i,:] == Vec([3*i + j + 1 for j in range(3)], orient="h")
        for j in range(3):
            assert m[:,j] == Vec([3*i + j + 1 for i in range(3)], orient="v")
        assert m[::2,1::2] == Mat([[2], [8]])  # select even rows and odd columns
        assert m.without[::2,1::2] == Mat([[4, 6]])  # don't select even rows and odd columns

    def test_setitem(self, /) -> None:
        m = Mat([[5*i+j for j in range(5)] for i in range(5)])
        m0 = m.copy()  # save for later
        m[0,0] = 12
        assert m[0,0] == 12
        m[0,:][0] = 7  # m[0,:] returns a standalone vector, not associated with m anymore
        assert m[0,0] == 12
        m[0,0] -= 12
        assert m[0,0] == 0
        m *= 2
        assert m == Mat([[10*i+2*j for j in range(5)] for i in range(5)])
        m -= (m/2)
        assert m == m0
        m[0,:] -= Vec(range(5), orient="h")
        assert not m[0,:]           # make sure the first row is now all 0s
        assert m[1:,:] == m0[1:,:]  # make sure no other row was changed
        m = m0.copy()
        m[0,:] *= 2
        assert m[0,:] == 2*m0[0,:]  # make sure the first row was doubled
        assert m[1:,:] == m0[1:,:]  # make sure no other row was changed
