# This file is part of https://github.com/KurtBoehm/lineal.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np


def _print_value(v: float, digits: int, width: int):
    form = f"{{:.{digits}f}}"

    if v == 0:
        return "·".center(width)
    if v == int(v):
        return str(int(v)).rjust(width)
    return form.format(v).rjust(width)


def mat_to_str(mat: np.ndarray, digits: int = 1, width: int = 3) -> str:
    assert len(mat.shape) == 2

    def print_value(v: float):
        return _print_value(v, digits, width)

    return "\n".join(" ".join(print_value(v) for v in r) for r in mat)


def vec_to_str(vec: np.ndarray, digits: int = 1, width: int = 3) -> str:
    assert len(vec.shape) == 1

    def print_value(v: float):
        return _print_value(v, digits, width)

    return "[" + " ".join(print_value(v) for v in vec) + "]"
