# This file is part of https://github.com/KurtBoehm/lineal.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np

from .defs import ProblemInfo, harmonic_mean


def diffusion_matrix(info: ProblemInfo, materials: list[int]):
    quotients = info.quotients
    flow_limits = (0, info.dims[info.flow_axis] - 1)

    def coeff(i: int, j: int) -> float:
        p1, p2 = info.index_to_pos(i), info.index_to_pos(j)
        dist = sum(abs(v1 - v2) for v1, v2 in zip(p1, p2))
        if dist == 0:
            if p1[info.flow_axis] in flow_limits:
                add = 2 * quotients[info.flow_axis] * info.diff_coeffs[materials[i]]
            else:
                add = 0.0
            value = sum(-coeff(i, k) for k in range(info.full_size) if i != k) + add
            return value if value > 0 else 1
        elif dist == 1:
            return -harmonic_mean(
                info.diff_coeffs[materials[i]], info.diff_coeffs[materials[j]]
            )
        else:
            return 0.0

    return np.array(
        [[coeff(i, j) for j in range(info.full_size)] for i in range(info.full_size)]
    )
