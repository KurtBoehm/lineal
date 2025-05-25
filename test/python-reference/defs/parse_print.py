# This file is part of https://github.com/KurtBoehm/lineal.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import numpy.typing as npt


def parse_print(txt: str) -> npt.NDArray[np.float64]:
    lines = txt.splitlines()
    lines = [lstrip for line in lines if len(lstrip := line.strip()) > 0]
    n = len(lines)
    mat = np.zeros((n, n))

    for line in lines:
        i, cols = line.split(": ")
        i = int(i)
        for j, v in (s.split(" → ") for s in cols.split(",")):
            mat[i, int(j)] = v

    return mat
