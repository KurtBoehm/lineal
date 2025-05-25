// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_AGGREGATION_MATRIX_AGGREGATION_HPP
#define INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_AGGREGATION_MATRIX_AGGREGATION_HPP

#include <cassert>
#include <cstddef>
#include <type_traits>

#include "lineal/base.hpp"
#include "lineal/linear-solver/multigrid/graph.hpp"

namespace lineal::amg {
template<typename TDependencyDetective, typename TAggregator, typename TMatTransform,
         typename TByteAlloc>
struct MatrixAggregator {
  using DependencyDetective = std::decay_t<TDependencyDetective>;
  using Aggregator = std::decay_t<TAggregator>;
  using MatrixTransform = TMatTransform;

  MatrixAggregator(TDependencyDetective&& dep_detective, TAggregator&& aggregator)
      : dep_detective_(std::forward<TDependencyDetective>(dep_detective)),
        aggregator_(std::forward<TAggregator>(aggregator)) {}

  auto aggregate_impl(const SharedMatrix auto& fine_mat, std::size_t level_idx, bool is_aggressive,
                      const auto& env) const {
    auto graph = make_matrix_graph<TMatTransform, TByteAlloc>(fine_mat, env.execution_policy());

    auto prop_graph = env.add_object("make_prop_graph", [&](const auto& envi) {
      return make_propertied_graph<TByteAlloc>(graph, dep_detective_, is_aggressive, envi);
    });

    auto agg_map = env.add_object("aggregate", [&](const auto& envi) {
      return aggregator_.aggregate(prop_graph, level_idx, is_aggressive, envi);
    });
    assert(fine_mat.row_num() == 0 || agg_map.coarse_row_num() > 0);

    return agg_map;
  }

  auto aggregate_matrix(const SharedMatrix auto& fine_mat, std::size_t level_idx,
                        bool is_aggressive, const auto& env) const {
    return aggregate_impl(fine_mat, level_idx, is_aggressive, env);
  }

private:
  TDependencyDetective dep_detective_;
  TAggregator aggregator_;
};
} // namespace lineal::amg

#endif // INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_AGGREGATION_MATRIX_AGGREGATION_HPP
