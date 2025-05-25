// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_DEPENDENCY_CRITERION_HPP
#define INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_DEPENDENCY_CRITERION_HPP

#include <algorithm>
#include <cassert>
#include <optional>
#include <tuple>
#include <type_traits>
#include <vector>

#include "thesauros/macropolis.hpp"

#include "lineal/base.hpp"

namespace lineal::amg {
template<typename TPropGraph>
struct DuneDependencyCriterionThreadInstance {
  using PropertiesGraph = TPropGraph;
  using ExtEdge = PropertiesGraph::ExtEdge;
  using Vertex = PropertiesGraph::Vertex;
  using FullVertex = PropertiesGraph::FullVertex;
  using WeightedGraph = PropertiesGraph::WeightedGraph;
  using GraphWeight = WeightedGraph::Weight;
  using Values = std::vector<std::tuple<ExtEdge, std::optional<ExtEdge>, GraphWeight>>;

  DuneDependencyCriterionThreadInstance(TPropGraph& graph, GraphWeight strong_threshold)
      : graph_(graph), strong_threshold_(strong_threshold) {}

  THES_ALWAYS_INLINE void handle(const FullVertex& vtx, auto memory_order) {
    const auto& wgraph = graph_.weighted_graph();

    values_.clear();
    GraphWeight vertex_weight = 0;
    GraphWeight max_strength = 0;

    vtx.iterate_ext([&](auto&& /*vtx*/, auto w) THES_ALWAYS_INLINE { vertex_weight = w; },
                    [&](auto&& e, auto w) THES_ALWAYS_INLINE {
                      const GraphWeight nn_weight = std::max(w, GraphWeight{0});
                      const GraphWeight opposite_vertex_weight = wgraph.head_vertex_weight(e);
                      std::optional<ExtEdge> opposite_edge = wgraph.reverse(e);

                      const GraphWeight opposite_edge_weight = [&]() THES_ALWAYS_INLINE {
                        if (wgraph.is_value_symmetric() == IsSymmetric{true}) {
                          return nn_weight;
                        }
                        return opposite_edge.has_value()
                                 ? std::max(wgraph.weight_of(*opposite_edge), GraphWeight{0})
                                 : GraphWeight{0};
                      }();
                      const GraphWeight strength =
                        nn_weight * opposite_edge_weight / opposite_vertex_weight;
                      max_strength = std::max(max_strength, strength);
                      values_.emplace_back(e, opposite_edge, strength);
                    },
                    valued_tag, unordered_tag);

    bool any_strong = false;

    const auto lb = strong_threshold_ * max_strength;
    for (const auto& [edge, opposite_edge, strength] : values_) {
      const bool cond = strength > lb;
      any_strong = any_strong || cond;
      graph_.set_depends(edge, cond, memory_order);
      if (opposite_edge.has_value()) [[likely]] {
        graph_.set_influences(*opposite_edge, cond, memory_order);
      }
    }

    const bool is_isolated = !any_strong;

    graph_.set_isolated(vtx.vertex(), is_isolated, memory_order);
  }

private:
  PropertiesGraph& graph_;
  GraphWeight strong_threshold_;
  Values values_{};
};

template<typename TReal>
struct DuneDependencyCriterion {
  using Real = TReal;

  template<typename TPropGraph>
  struct Instance {
    using PropertiesGraph = TPropGraph;
    using WeightedGraph = PropertiesGraph::WeightedGraph;
    using GraphWeight = WeightedGraph::Weight;
    using ThreadInstance = DuneDependencyCriterionThreadInstance<TPropGraph>;

    Instance(TPropGraph& graph, GraphWeight strong_threshold)
        : graph_(graph), strong_threshold_(strong_threshold) {}

    auto thread_instance() {
      return ThreadInstance{graph_, strong_threshold_};
    }

  private:
    PropertiesGraph& graph_;
    GraphWeight strong_threshold_;
  };

  DuneDependencyCriterion(Real strong_threshold, Real aggressive_strong_threshold)
      : strong_threshold_(strong_threshold),
        aggressive_strong_threshold_(aggressive_strong_threshold) {}

  THES_ALWAYS_INLINE Real strong_threshold(bool is_aggressive) const {
    return is_aggressive ? aggressive_strong_threshold_ : strong_threshold_;
  }
  template<typename TPropGraph>
  Instance<TPropGraph> instantiate(TPropGraph& pg, bool is_aggressive) const {
    using GraphWeight = std::decay_t<TPropGraph>::WeightedGraph::Weight;
    return {pg, GraphWeight(strong_threshold(is_aggressive))};
  }

private:
  Real strong_threshold_;
  Real aggressive_strong_threshold_;
};
} // namespace lineal::amg

#endif // INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_DEPENDENCY_CRITERION_HPP
