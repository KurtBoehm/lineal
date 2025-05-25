# This file is part of https://github.com/KurtBoehm/tlaxcaltin.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

project(
  'expected',
  'cpp',
  version: '1.1',
  license: 'CC0-1.0',
)

install_headers(['include/tl/expected.hpp'], subdir: 'tl')

expected_dep = declare_dependency(
  include_directories: include_directories('include'),
)
