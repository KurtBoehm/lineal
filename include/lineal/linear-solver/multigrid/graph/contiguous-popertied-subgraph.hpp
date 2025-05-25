// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_GRAPH_CONTIGUOUS_POPERTIED_SUBGRAPH_HPP
#define INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_GRAPH_CONTIGUOUS_POPERTIED_SUBGRAPH_HPP

#include <cassert>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "thesauros/format.hpp"
#include "thesauros/iterator.hpp"
#include "thesauros/macropolis.hpp"
#include "thesauros/types.hpp"

#include "lineal/base.hpp"
#include "lineal/component-wise/exec-constraint.hpp"
#include "lineal/parallel/index.hpp"

namespace lineal::amg {
template<SharedGraph TPropGraph>
struct ContiguousPropertiedSubgraph : public SharedGraphBase {
  using Self = ContiguousPropertiedSubgraph;
  using PropertiedGraph = std::decay_t<TPropGraph>;
  using Size = PropertiedGraph::Size;
  using Vertex = PropertiedGraph::Vertex;
  using VertexSize = PropertiedGraph::VertexSize;
  using VertexIdx = PropertiedGraph::VertexIdx;
  using VertexRange = IndexRange<VertexIdx>;
  using Edge = PropertiedGraph::Edge;
  using EdgeSize = PropertiedGraph::EdgeSize;

  using PropGraphFullVertex = PropertiedGraph::FullVertex;
  using PropGraphIter = PropertiedGraph::const_iterator;
  using PropGraphDiff = std::iterator_traits<PropGraphIter>::difference_type;

  using EdgeProperties = PropertiedGraph::EdgeProperties;

  static constexpr auto exec_constraints = lineal::exec_constraints<PropertiedGraph>;

  struct FullVertex {
    explicit FullVertex(VertexRange vtx_range, PropGraphFullVertex vtx)
        : range_(vtx_range), vertex_(std::move(vtx)) {}

    VertexIdx index() const {
      return vertex_.index();
    }
    Vertex vertex() const {
      return vertex_.vertex();
    }
    VertexSize full_adjacent_num() const {
      return vertex_.full_adjacent_num();
    }

    THES_ALWAYS_INLINE constexpr decltype(auto) iterate(auto vtx_op, auto edge_op,
                                                        AnyValuationTag auto is_valued,
                                                        AnyOrderingTag auto is_ordered) const {
      assert(range_.contains(index()));

      return vertex_.iterate(
        std::move(vtx_op),
        [&](auto edge, auto... args) THES_ALWAYS_INLINE {
          using Ret = decltype(edge_op(std::move(edge), args...));

          if (range_.contains(edge.head().index())) {
            return edge_op(std::move(edge), args...);
          }
          if constexpr (!std::is_void_v<Ret>) {
            return Ret{};
          }
        },
        is_valued, is_ordered);
    }

  private:
    VertexRange range_;
    PropGraphFullVertex vertex_;
  };

private:
  struct ConstIterProvider {
    using Value = FullVertex;
    struct IterTypes : public thes::iter_provider::ValueTypes<Value, PropGraphDiff> {
      using IterState = PropGraphIter;
    };

    static Value deref(const auto& self) {
      return FullVertex(self.range_, *self.iter_);
    }
    template<typename TSelf>
    static thes::TransferConst<TSelf, PropGraphIter>& state(TSelf& self) {
      return self.iter_;
    }
  };

public:
  struct const_iterator
      : public thes::IteratorFacade<const_iterator, thes::iter_provider::Map<ConstIterProvider>> {
    friend struct ConstIterProvider;
    explicit const_iterator(VertexRange vtx_range, PropGraphIter iter)
        : range_(vtx_range), iter_(std::move(iter)) {}

  private:
    VertexRange range_;
    PropGraphIter iter_;
  };

  // TODO Is this base appropriate?
  struct FullView : public SharedGraphBase {
    using Size = Self::Size;
    using const_iterator = Self::const_iterator;
    static constexpr auto exec_constraints = Self::exec_constraints;

    explicit FullView(const Self& range) : self_(&range) {}

    [[nodiscard]] const_iterator begin() const {
      return self_->full_begin();
    }
    [[nodiscard]] const_iterator end() const {
      return self_->full_end();
    }

  private:
    const Self* self_;
  };

  ContiguousPropertiedSubgraph(TPropGraph&& prop_graph, VertexRange vtx_range)
      : properties_graph_(std::forward<TPropGraph>(prop_graph)), range_(vtx_range) {
    const auto end_idx = index_value(range_.end_index());
    if (end_idx > properties_graph_.vertex_num()) {
      throw std::invalid_argument{
        fmt::format("The graph has {} < {} vertices", properties_graph_.vertex_num(), end_idx)};
    }
  }

  ContiguousPropertiedSubgraph(const ContiguousPropertiedSubgraph&) = default;
  ContiguousPropertiedSubgraph& operator=(const ContiguousPropertiedSubgraph&) = default;
  ContiguousPropertiedSubgraph(ContiguousPropertiedSubgraph&&) noexcept = default;
  ContiguousPropertiedSubgraph& operator=(ContiguousPropertiedSubgraph&&) noexcept = default;
  ~ContiguousPropertiedSubgraph() = default;

  const_iterator begin() const {
    return const_iterator{range_, properties_graph_.begin() + vertex_index_begin()};
  }
  const_iterator end() const {
    return const_iterator{range_, properties_graph_.begin() + vertex_index_end()};
  }

  const_iterator full_begin() const {
    return const_iterator(range_, properties_graph_.begin());
  }
  const_iterator full_end() const {
    return const_iterator(range_, properties_graph_.end());
  }
  FullView full_view() const {
    return FullView(*this);
  }
  [[nodiscard]] VertexSize full_vertex_num() const {
    return properties_graph_.vertex_num();
  }
  [[nodiscard]] VertexSize vertex_num() const {
    return range_.size();
  }

  [[nodiscard]] const PropertiedGraph& full_graph() const {
    return properties_graph_;
  }

#define LINEAL_COPY_MEMBER(NAME, QUALIFIER) \
  template<typename... TArgs> \
  THES_ALWAYS_INLINE decltype(auto) NAME(TArgs&&... args) QUALIFIER \
  requires(requires(QUALIFIER TPropGraph& g) { g.NAME(std::forward<TArgs>(args)...); }) \
  { \
    return properties_graph_.NAME(std::forward<TArgs>(args)...); \
  }

  LINEAL_COPY_MEMBER(is_isolated, const)
  LINEAL_COPY_MEMBER(set_isolated, )

  LINEAL_COPY_MEMBER(set_edge_properties, )
  LINEAL_COPY_MEMBER(get_edge_properties, const)

  LINEAL_COPY_MEMBER(max_edge_num, const)
  LINEAL_COPY_MEMBER(max_edge_index_end, const)

  LINEAL_COPY_MEMBER(reverse, const)

  [[nodiscard]] VertexIdx vertex_index_begin() const {
    return range_.begin_index();
  }
  [[nodiscard]] VertexIdx vertex_index_end() const {
    return range_.end_index();
  }

  THES_ALWAYS_INLINE FullVertex full_vertex_at(VertexIdx index) const {
    return FullVertex{range_, properties_graph_.full_vertex_at(index)};
  }
  THES_ALWAYS_INLINE FullVertex head_full_vertex(const Edge& edge) const {
    return FullVertex{range_, properties_graph_.head_full_vertex(edge)};
  }

#undef LINEAL_COPY_MEMBER

private:
  TPropGraph properties_graph_;
  VertexRange range_;
};
} // namespace lineal::amg

#endif // INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_GRAPH_CONTIGUOUS_POPERTIED_SUBGRAPH_HPP
