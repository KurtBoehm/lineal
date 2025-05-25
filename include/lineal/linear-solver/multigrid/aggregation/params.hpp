// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_AGGREGATION_PARAMS_HPP
#define INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_AGGREGATION_PARAMS_HPP

#include <optional>

namespace lineal::amg {
template<typename TSize>
struct AggregationParams {
  using Size = TSize;

  Size min_aggregate_size = 6;
  Size max_aggregate_size = 9;
  std::optional<Size> min_per_thread{};
};
} // namespace lineal::amg

#endif // INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_AGGREGATION_PARAMS_HPP
