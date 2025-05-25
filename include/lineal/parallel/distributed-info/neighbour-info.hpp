// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_PARALLEL_DISTRIBUTED_INFO_NEIGHBOUR_INFO_HPP
#define INCLUDE_LINEAL_PARALLEL_DISTRIBUTED_INFO_NEIGHBOUR_INFO_HPP

#include <type_traits>
#include <utility>

#include "lineal/base.hpp"

namespace lineal {
template<typename TBefore, typename TAfter>
requires(!std::is_reference_v<TBefore> && !std::is_reference_v<TAfter>)
struct NeighbourInfo {
  NeighbourInfo()
  requires(std::is_default_constructible_v<TBefore> && std::is_default_constructible_v<TAfter>)
  = default;

  NeighbourInfo(TBefore&& before, TAfter&& after)
      : before_(std::move(before)), after_(std::move(after)) {}

  [[nodiscard]] const TBefore& get(BeforeTag /*tag*/) const {
    return before_;
  }
  [[nodiscard]] TBefore& get(BeforeTag /*tag*/) {
    return before_;
  }

  [[nodiscard]] const TAfter& get(AfterTag /*tag*/) const {
    return after_;
  }
  [[nodiscard]] TAfter& get(AfterTag /*tag*/) {
    return after_;
  }

private:
  TBefore before_{};
  TAfter after_{};
};
} // namespace lineal

#endif // INCLUDE_LINEAL_PARALLEL_DISTRIBUTED_INFO_NEIGHBOUR_INFO_HPP
