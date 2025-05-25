// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_PARALLEL_DISTRIBUTED_INFO_OPERATION_HPP
#define INCLUDE_LINEAL_PARALLEL_DISTRIBUTED_INFO_OPERATION_HPP

#include "thesauros/utility.hpp"

namespace lineal {
inline constexpr decltype(auto) distributed_info_storage(const auto& something) {
  if constexpr (requires { something.distributed_info(); }) {
    return something.distributed_info();
  } else {
    return thes::Empty{};
  }
}
} // namespace lineal

#endif // INCLUDE_LINEAL_PARALLEL_DISTRIBUTED_INFO_OPERATION_HPP
