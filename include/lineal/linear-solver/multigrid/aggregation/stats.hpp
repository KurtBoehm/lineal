// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_AGGREGATION_STATS_HPP
#define INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_AGGREGATION_STATS_HPP

#include <atomic>
#include <cstddef>

#include "thesauros/macropolis.hpp"

namespace lineal {
struct AggregationStats {
  THES_DEFINE_TYPE(SNAKE_CASE(AggregationStats), NO_CONSTRUCTOR,
                   MEMBERS((KEEP(aggregate_num), std::size_t, 0),
                           (KEEP(multi_vertex_aggregate_num), std::size_t, 0),
                           (KEEP(single_vertex_aggregate_num), std::size_t, 0),
                           (KEEP(merged_aggregate_num), std::size_t, 0),
                           (KEEP(isolated_aggregate_num), std::size_t, 0)))

  AggregationStats& operator+=(const AggregationStats& other) {
    std::atomic_ref{aggregate_num} += other.aggregate_num;
    std::atomic_ref{multi_vertex_aggregate_num} += other.multi_vertex_aggregate_num;
    std::atomic_ref{single_vertex_aggregate_num} += other.single_vertex_aggregate_num;
    std::atomic_ref{merged_aggregate_num} += other.merged_aggregate_num;
    std::atomic_ref{isolated_aggregate_num} += other.isolated_aggregate_num;
    return *this;
  }
};
} // namespace lineal

#endif // INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_AGGREGATION_STATS_HPP
