# This file is part of https://github.com/KurtBoehm/lineal.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from pathlib import Path

import numpy as np

base_path = Path(__file__).parent.resolve()
with open(base_path / "input.txt", "r") as f:
    lines = [l for line in f if len(l := line.strip()) > 0]
    n = len(lines)
    mat = np.zeros((n, n))
    for l in lines:
        i, cols = l.split(": ")
        for p in cols.split(", "):
            j, v = p.split(" → ")
            mat[int(i), int(j)] = float(v)
    print(f"{n}×{n} matrix; rank: {np.linalg.matrix_rank(mat)}")
