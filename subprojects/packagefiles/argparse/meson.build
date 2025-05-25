# This file is part of https://github.com/KurtBoehm/tlaxcaltin.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

project(
  'argparse',
  'cpp',
  version: '3.2',
  license: 'MIT',
)

install_headers(
  ['include/argparse/argparse.hpp'],
  subdir: 'argparse',
)

argparse_dep = declare_dependency(
  include_directories: include_directories('include'),
)
