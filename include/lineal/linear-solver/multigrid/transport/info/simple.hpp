// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_TRANSPORT_INFO_SIMPLE_HPP
#define INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_TRANSPORT_INFO_SIMPLE_HPP

#include "lineal/base/concept/transport-info.hpp"
#include "lineal/linear-solver/multigrid/aggregate-map.hpp"

namespace lineal {
template<typename TSizeByte, typename TByteAlloc>
struct SimpleTransportInfo : public SharedTransportInfoBase {
  using AggregateMap = amg::MultithreadAggregateMap<TSizeByte, TByteAlloc>;

  explicit SimpleTransportInfo(AggregateMap&& agg_map)
      : agg_map_(std::forward<AggregateMap>(agg_map)) {}

  [[nodiscard]] const AggregateMap& aggregate_map() const {
    return agg_map_;
  }

private:
  AggregateMap agg_map_;
};
} // namespace lineal

#endif // INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_TRANSPORT_INFO_SIMPLE_HPP
