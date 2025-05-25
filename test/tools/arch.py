# This file is part of https://github.com/KurtBoehm/lineal.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from subprocess import PIPE, run


def parse_line(line: str):
    line = line.strip()
    assert len(line) > 0
    lhs, rhs = line.split(":")
    return lhs.strip(), rhs.strip()


arch = run(["uname", "--machine"], stdout=PIPE).stdout.decode().strip()
assert arch == "x86_64"


x86_64_levels = {
    "x86-64": ["cmov", "cx8", "fpu", "fxsr", "mmx", "syscall", "sse", "sse2"],
    "x86-64-v2": ["cx16", "lahf_lm", "popcnt", "sse4_1", "sse4_2", "ssse3"],
    "x86-64-v3": [
        "avx",
        "avx2",
        "bmi1",
        "bmi2",
        "f16c",
        "fma",
        "abm",
        "movbe",
        "xsave",
    ],
    "x86-64-v4": ["avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl"],
}


with open("/proc/cpuinfo", "r") as f:
    raw_info = f.read()
info = {
    k: v
    for k, v in (parse_line(l) for l in raw_info.splitlines() if len(l.strip()) != 0)
}
# print(info)
flags = {f.strip() for f in info["flags"].split()}
# print(flags)
supported = [k for k, v in x86_64_levels.items() if all(f in flags for f in v)]
print(" ".join(supported))
