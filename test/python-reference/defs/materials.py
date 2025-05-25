# This file is part of https://github.com/KurtBoehm/lineal.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from collections.abc import Sequence

from .defs import ProblemInfo

CellInfos = list[int]


def from_centre(p: Sequence[int], dims: Sequence[int]) -> Sequence[float]:
    return [(arg1 / arg2 - 0.5) ** 2 for arg1, arg2 in zip(p, dims)]


def from_centre_sum(p1: Sequence[int], p2: Sequence[int]) -> float:
    return sum(from_centre(p1, p2))


def flow_cylinder_gap_material(info: ProblemInfo, radius: float) -> CellInfos:
    flow_axis = info.flow_axis
    size = info.full_size
    material = [0] * size

    mid = info.dims[flow_axis] // 2
    nnz = 0
    for index in range(size):
        pos = info.index_to_pos(index)
        value = from_centre_sum(
            pos.all_except(flow_axis), info.dims.all_except(flow_axis)
        )
        if value <= radius and pos[flow_axis] != mid:
            nnz += 1
            material[index] = 3

    return material
