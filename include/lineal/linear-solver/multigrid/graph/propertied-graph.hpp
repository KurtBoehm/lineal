// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_GRAPH_PROPERTIED_GRAPH_HPP
#define INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_GRAPH_PROPERTIED_GRAPH_HPP

#include <atomic>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

#include "thesauros/containers.hpp"
#include "thesauros/memory.hpp"
#include "thesauros/types.hpp"
#include "thesauros/utility.hpp"

#include "lineal/base.hpp"
#include "lineal/linear-solver/multigrid/graph/properties.hpp"
#include "lineal/parallel.hpp"

namespace lineal::amg {
// A bourgeois graph exploiting the proletariat.
template<AnyGraph TWeightedGraph, typename TByteAlloc = thes::HugePagesAllocator<std::byte>>
struct PropertiedGraph : public SharedGraphBase {
  using WeightedGraph = std::decay_t<TWeightedGraph>;
  using Matrix = WeightedGraph::Matrix;
  using Size = WeightedGraph::Size;
  using DistributedInfo = DistributedInfoOf<WeightedGraph>;
  using DistributedInfoStorage = thes::VoidStorage<DistributedInfo>;
  static constexpr bool is_shared = std::is_void_v<DistributedInfo>;

  using Chunk = thes::u8;
  using Alloc = std::allocator_traits<TByteAlloc>::template rebind_alloc<Chunk>;

  using Vertex = WeightedGraph::Vertex;
  using FullVertex = WeightedGraph::FullVertex;
  using VertexSize = WeightedGraph::VertexSize;
  using VertexIdx = WeightedGraph::VertexIdx;
  using Edge = WeightedGraph::Edge;
  using ExtEdge = WeightedGraph::ExtEdge;
  using EdgeSize = WeightedGraph::EdgeSize;
  using const_iterator = WeightedGraph::const_iterator;

  using EdgeProperties = amg::EdgeProperties;

  static constexpr auto exec_constraints = lineal::exec_constraints<WeightedGraph>;

  PropertiedGraph(TWeightedGraph&& graph, const auto& dependency_detective, bool is_aggressive,
                  const Env auto& env)
      : weighted_(std::forward<TWeightedGraph>(graph)),
        vertex_isolated_(index_value(weighted_.vertex_index_end(), own_index_tag,
                                     distributed_info_storage(weighted_))),
        edge_properties_(weighted_.max_edge_index_end()) {
    dependency_detective.build(*this, is_aggressive, env);
  }

  PropertiedGraph(const PropertiedGraph&) = delete;
  PropertiedGraph& operator=(const PropertiedGraph&) = delete;
  PropertiedGraph(PropertiedGraph&&) noexcept = default;
  PropertiedGraph& operator=(PropertiedGraph&&) noexcept = default;
  ~PropertiedGraph() = default;

  const_iterator begin() const {
    return weighted_.begin();
  }
  const_iterator end() const {
    return weighted_.end();
  }

  void set_isolated(const Vertex& vertex, bool isolated, thes::Empty /*mem_order*/ = {}) {
    vertex_isolated_[index_value(vertex.index())] = isolated;
  }
  void set_isolated(const Vertex& vertex, bool isolated, std::memory_order mem_order) {
    vertex_isolated_[index_value(vertex.index())].store(isolated, mem_order);
  }

  [[nodiscard]] bool is_isolated(const Vertex& vertex) const {
    return vertex_isolated_[index_value(vertex.index())];
  }

  void set_edge_properties(const ExtEdge& edge, bool depends, bool influences,
                           thes::Empty /*mem_order*/ = {}) {
    edge.index().value_run([&](auto edge_idx) {
      edge_properties_[edge_idx] = EdgeProperties{depends, influences}.to_byte();
    });
  }
  void set_edge_properties(const ExtEdge& edge, bool depends, bool influences,
                           std::memory_order order) {
    edge.index().value_run([&](auto edge_idx) {
      edge_properties_[edge_idx].store(EdgeProperties{depends, influences}.to_byte(), order);
    });
  }

  void set_depends(const ExtEdge& edge, bool value, thes::Empty /*mem_order*/ = {}) {
    edge.index().value_run([&](auto edge_idx) {
      edge_properties_[edge_idx].set_bit(EdgeProperties::depends_bit, value);
    });
  }
  void set_depends(const ExtEdge& edge, bool value, std::memory_order order) {
    edge.index().value_run([&](auto edge_idx) {
      edge_properties_[edge_idx].set_bit(EdgeProperties::depends_bit, value, order);
    });
  }

  void set_influences(const ExtEdge& edge, bool value, thes::Empty /*mem_order*/ = {}) {
    edge.index().value_run([&](auto edge_idx) {
      edge_properties_[edge_idx].set_bit(EdgeProperties::influences_bit, value);
    });
  }
  void set_influences(const ExtEdge& edge, bool value, std::memory_order order) {
    edge.index().value_run([&](auto edge_idx) {
      edge_properties_[edge_idx].set_bit(EdgeProperties::influences_bit, value, order);
    });
  }

  [[nodiscard]] EdgeProperties get_edge_properties(const Edge& edge) const {
    return EdgeProperties::from_byte(edge_properties_[edge.index().value()]);
  }

  [[nodiscard]] const WeightedGraph& weighted_graph() const {
    return weighted_;
  }
  [[nodiscard]] const Matrix& matrix() const {
    return weighted_.matrix();
  }

  [[nodiscard]] VertexSize vertex_num() const {
    return weighted_.vertex_num();
  }
  [[nodiscard]] VertexIdx vertex_index_begin() const {
    return weighted_.vertex_index_begin();
  }
  [[nodiscard]] VertexIdx vertex_index_end() const {
    return weighted_.vertex_index_end();
  }

  [[nodiscard]] EdgeSize max_edge_num() const {
    return weighted_.max_edge_num();
  }
  [[nodiscard]] EdgeSize max_edge_index_end() const {
    return weighted_.max_edge_index_end();
  }

  FullVertex full_vertex_at(auto index) const
  requires(requires(const WeightedGraph& weighted) { weighted.full_vertex_at(index); })
  {
    return weighted_.full_vertex_at(index);
  }

  const DistributedInfoStorage& distributed_info() const
  requires(!is_shared)
  {
    return weighted_.distributed_info();
  }

private:
  TWeightedGraph weighted_;
  thes::DynamicBitset<Chunk, Alloc> vertex_isolated_;
  thes::MultiBitIntegers<Chunk, 2, Alloc> edge_properties_;
};
template<typename TByteAlloc, AnyGraph TWeightedGraph>
inline PropertiedGraph<TWeightedGraph, TByteAlloc>
make_propertied_graph(TWeightedGraph&& graph, const auto& dependency_detective, bool is_aggressive,
                      const Env auto& env) {
  return PropertiedGraph<TWeightedGraph, TByteAlloc>{std::forward<TWeightedGraph>(graph),
                                                     dependency_detective, is_aggressive, env};
}
} // namespace lineal::amg

#endif // INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_GRAPH_PROPERTIED_GRAPH_HPP
