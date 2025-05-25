# This file is part of https://github.com/KurtBoehm/lineal.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import NamedTuple


class Int2d(NamedTuple):
    x: int
    y: int


class Int3d(NamedTuple):
    x: int
    y: int
    z: int

    def all_except(self, index: int) -> Int2d:
        return Int2d(*(v for i, v in enumerate(self) if i != index))


class Float2d(NamedTuple):
    x: float
    y: float


class Float3d(NamedTuple):
    x: float
    y: float
    z: float

    def all_except(self, index: int) -> Float2d:
        return Float2d(*(v for i, v in enumerate(self) if i != index))


@dataclass
class ProblemInfo:
    dims: Int3d
    lengths: Float3d
    diff_coeffs: list[float]
    flow_axis: int = 0

    @property
    def full_size(self) -> int:
        return reduce(mul, self.dims)

    @property
    def cell_lengths(self) -> Float3d:
        return Float3d(
            *(length / size for length, size in zip(self.lengths, self.dims))
        )

    @property
    def quotients(self) -> Float3d:
        cell_lengths = self.cell_lengths
        return Float3d(
            *(
                reduce(mul, cell_lengths.all_except(i)) / cell_lengths[i]
                for i in range(3)
            )
        )

    def index_to_pos(self, index: int) -> Int3d:
        dim_x, dim_y, dim_z = self.dims
        z = index % dim_z
        div = index // dim_z
        y = div % dim_y
        x = div // dim_y
        assert x < dim_x
        return Int3d(x, y, z)


def harmonic_mean(a: float, b: float) -> float:
    p = a * b
    return 2 * p / (a + b) if p != 0 else 0
