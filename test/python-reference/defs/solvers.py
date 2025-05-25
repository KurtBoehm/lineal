# This file is part of https://github.com/KurtBoehm/lineal.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np


def sor(relax: float, lhs: np.ndarray, sol: np.ndarray, rhs: np.ndarray):
    n = len(sol)
    for i in range(n):
        sol[i] = (1 - relax) * sol[i] + relax / lhs[i, i] * (
            rhs[i] - sum(lhs[i, j] * sol[j] for j in range(n) if i != j)
        )
