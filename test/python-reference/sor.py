# This file is part of https://github.com/KurtBoehm/lineal.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from json import dumps

import numpy as np
from defs import (
    Float3d,
    Int3d,
    ProblemInfo,
    diffusion_matrix,
    flow_cylinder_gap_material,
    sor,
)


def print_vector(vec: list[float] | list[int] | np.ndarray):
    if not isinstance(vec, list):
        vec = list(vec)
    print(dumps(vec))


info = ProblemInfo(
    dims=Int3d(4, 8, 4),
    lengths=Float3d(4.0, 8.0, 4.0),
    diff_coeffs=[0.0, 1.0, 2.0, 3.0],
)
radius = 2.0
relax = 1.0

materials = flow_cylinder_gap_material(info, radius)
# print_vector(materials)
lhs = diffusion_matrix(info, materials)
# print(lhs)
vec = np.array([float(i + 1) for i in range(info.full_size)])
rhs = lhs @ vec
# print_vector(vec)
# print_vector(rhs)

sol = np.zeros(info.full_size)
print_vector(sol)
# print(np.linalg.norm(lhs @ sol - rhs))
for i in range(128):
    sor(relax, lhs, sol, rhs)
    print_vector(sol)
    # print(np.linalg.norm(lhs @ sol - rhs))
