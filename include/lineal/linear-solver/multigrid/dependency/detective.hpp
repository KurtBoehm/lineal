// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_DEPENDENCY_DETECTIVE_HPP
#define INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_DEPENDENCY_DETECTIVE_HPP

#include <atomic>
#include <type_traits>
#include <utility>

#include "thesauros/macropolis.hpp"
#include "thesauros/types.hpp"

#include "lineal/component-wise/facade.hpp"
#include "lineal/vectorization.hpp"

namespace lineal {
struct DependencyDetectiveSinkConf {
  using Work = void;
  using Value = void;
  static constexpr bool supports_const_access = false;
  static constexpr bool supports_mutable_access = true;
  static constexpr bool custom_range = true;
};

template<typename TPropertiesGraph, typename TCriterionInstance>
struct DependencyDetectiveThreadInstance
    : public facades::ComponentWiseOp<
        DependencyDetectiveThreadInstance<TPropertiesGraph, TCriterionInstance>,
        DependencyDetectiveSinkConf, TPropertiesGraph&> {
  using CriterionInstance = std::decay_t<TCriterionInstance>;
  using PropertiesGraph = TPropertiesGraph;
  using Weight = PropertiesGraph::WeightedGraph::Weight;

  using Parent = facades::ComponentWiseOp<DependencyDetectiveThreadInstance,
                                          DependencyDetectiveSinkConf, TPropertiesGraph&>;

  DependencyDetectiveThreadInstance(TPropertiesGraph& prop_graph,
                                    TCriterionInstance&& criterion_instance)
      : Parent(prop_graph),
        criterion_instance_{std::forward<TCriterionInstance>(criterion_instance)} {}

  void compute_base(grex::Scalar /*tag*/, auto&& vertex) {
    criterion_instance_.handle(vertex, std::memory_order_relaxed);
  }
  THES_ALWAYS_INLINE void compute_impl(grex::Scalar tag, const auto& /*children*/, auto vtx_it) {
    compute_base(tag, *vtx_it);
  };
  THES_ALWAYS_INLINE void compute_impl(auto tag, const auto& arg, const auto& /*children*/,
                                       const auto& graph) {
    compute_base(tag, graph.full_vertex_at(arg));
  }

  THES_ALWAYS_INLINE auto begin_impl(thes::IndexTag<0> /*tag*/, auto& graph) {
    return std::begin(graph);
  }
  THES_ALWAYS_INLINE auto end_impl(thes::IndexTag<0> /*tag*/, auto& graph) {
    return std::end(graph);
  }
  THES_ALWAYS_INLINE auto size() const {
    return prop_graph().vertex_num();
  }

private:
  [[nodiscard]] const PropertiesGraph& prop_graph() const {
    return thes::star::get_at<0>(this->children());
  }
  [[nodiscard]] const auto& matrix() const {
    return prop_graph().matrix();
  }

  TCriterionInstance criterion_instance_;
};

template<typename TCriterion>
struct DependencyDetective {
  using Criterion = std::decay_t<TCriterion>;

  template<typename TPropertiesGraph>
  struct Sink {
    using TCriterionInstance = Criterion::template Instance<TPropertiesGraph>;
    using CriterionThreadInstance = TCriterionInstance::ThreadInstance;
    using ThreadInstance =
      DependencyDetectiveThreadInstance<TPropertiesGraph, CriterionThreadInstance>;
    using Size = ThreadInstance::Size;
    static constexpr bool is_shared = ThreadInstance::is_shared;
    static constexpr auto exec_constraints = lineal::exec_constraints<TPropertiesGraph>;

    auto thread_instance(const auto& /*info*/, grex::Scalar /*tag*/) {
      return ThreadInstance{prop_graph_, criterion_instance_.thread_instance()};
    }

    Sink(TPropertiesGraph& prop_graph, TCriterionInstance criterion_instance)
        : prop_graph_{prop_graph},
          criterion_instance_{std::forward<TCriterionInstance>(criterion_instance)} {}

    THES_ALWAYS_INLINE auto size() const {
      return prop_graph_.vertex_num();
    }

  private:
    TPropertiesGraph& prop_graph_;
    TCriterionInstance criterion_instance_;
  };

  explicit DependencyDetective(TCriterion&& criterion)
      : criterion_{std::forward<TCriterion>(criterion)} {}

  template<typename TPropertiesGraph>
  void build(TPropertiesGraph& graph, bool is_aggressive, const auto& env) const {
    env.execution_policy().execute(Sink<TPropertiesGraph>{
      graph,
      criterion_.instantiate(graph, is_aggressive),
    });
  }

private:
  TCriterion criterion_;
};
} // namespace lineal

#endif // INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_DEPENDENCY_DETECTIVE_HPP
