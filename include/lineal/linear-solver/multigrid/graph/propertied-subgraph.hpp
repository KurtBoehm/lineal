// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_GRAPH_PROPERTIED_SUBGRAPH_HPP
#define INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_GRAPH_PROPERTIED_SUBGRAPH_HPP

#include <concepts>
#include <type_traits>

#include "thesauros/ranges.hpp"

#include "lineal/linear-solver/multigrid/graph/contiguous-popertied-subgraph.hpp"

namespace lineal::amg {
template<typename TPropGraph, typename TRange>
struct PropertiedSubgraphTrait;
template<typename TPropGraph, typename TSize>
struct PropertiedSubgraphTrait<TPropGraph, thes::IotaRange<TSize>> {
  static_assert(std::same_as<TSize, typename std::decay_t<TPropGraph>::Size>);
  using Type = ContiguousPropertiedSubgraph<TPropGraph>;
};

template<typename TPropGraph, typename TRange>
using PropertiedSubgraph = PropertiedSubgraphTrait<TPropGraph, TRange>::Type;
} // namespace lineal::amg

#endif // INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_GRAPH_PROPERTIED_SUBGRAPH_HPP
