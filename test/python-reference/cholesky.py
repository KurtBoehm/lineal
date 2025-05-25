# This file is part of https://github.com/KurtBoehm/lineal.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
from defs import mat_to_str, vec_to_str
from numpy.linalg import cholesky, solve

n = 32
lhs = np.zeros((n, n))
for i in range(n):
    if i > 0:
        lhs[i, i - 1] = -i
    lhs[i, i] = 2 * (i + 1)
    if i + 1 < n:
        lhs[i, i + 1] = -(i + 1)

print(mat_to_str(lhs, width=4))
print("*" * (n * 5))

chol = cholesky(lhs)
print(mat_to_str(chol, width=4))

rhs = lhs @ np.ones(n)
print(f"rhs: {vec_to_str(rhs)}")

aux = solve(chol, rhs)
print(f"aux: {vec_to_str(aux)}")

sol = solve(chol.T, aux)
print(f"sol: {vec_to_str(sol)}")
