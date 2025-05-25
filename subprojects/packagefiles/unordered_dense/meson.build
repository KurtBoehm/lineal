# This file is part of https://github.com/KurtBoehm/tlaxcaltin.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

project('unordered-dense', 'cpp', version: '4.5.0', license: 'MIT')

install_headers(['include/ankerl/unordered_dense.h'], subdir: 'ankerl')

unordered_dense_dep = declare_dependency(
  include_directories: include_directories('include'),
)
