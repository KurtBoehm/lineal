// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_PARALLEL_DISTRIBUTED_INFO_OPERATION_HPP
#define INCLUDE_LINEAL_PARALLEL_DISTRIBUTED_INFO_OPERATION_HPP

#include <cassert>

#include "thesauros/static-ranges.hpp"
#include "thesauros/utility.hpp"

namespace lineal {
constexpr decltype(auto) distributed_info_storage(const auto& something) {
  if constexpr (requires { something.distributed_info(); }) {
    return something.distributed_info();
  } else {
    return thes::Empty{};
  }
}
template<typename... Ts>
constexpr decltype(auto) unique_distributed_info_storage(const Ts&... something) {
  if constexpr ((... || requires { something.distributed_info(); })) {
    auto opt_dist_info =
      std::make_tuple(&something.distributed_info()...) | thes::star::unique_value;
    assert(opt_dist_info.has_value());
    return **opt_dist_info;
  } else {
    return thes::Empty{};
  }
}
} // namespace lineal

#endif // INCLUDE_LINEAL_PARALLEL_DISTRIBUTED_INFO_OPERATION_HPP
