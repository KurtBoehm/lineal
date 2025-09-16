// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_AGGREGATION_BASE_HPP
#define INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_AGGREGATION_BASE_HPP

#include <cassert>
#include <compare>
#include <cstddef>
#include <utility>
#include <vector>

#include "thesauros/containers.hpp"
#include "thesauros/functional.hpp"
#include "thesauros/macropolis.hpp"
#include "thesauros/static-ranges.hpp"
#include "thesauros/types.hpp"
#include "thesauros/utility.hpp"

#include "lineal/base.hpp"
#include "lineal/linear-solver/multigrid/aggregation/stats.hpp"
#include "lineal/parallel/index/index.hpp"

namespace lineal {
template<typename TVertexIdx>
struct AggregateFront {
  using VertexIdx = TVertexIdx;
  using VertexSize = IndexSize<VertexIdx>;

  AggregateFront(VertexIdx begin, VertexIdx end)
      : offset_{index_value(begin)}, size_{index_value(end) - index_value(begin)},
        front_markers_(size_, false) {}

  void clear() {
    for (const VertexIdx vtx : front_) {
      set(vtx, false);
    }
    front_.clear();
  }

  void erase(VertexIdx vtx) {
    front_.erase(vtx);
    set(vtx, false);
  }

  void insert(VertexIdx vtx) {
    front_.insert(vtx);
    set(vtx, true);
  }

  bool contains(VertexIdx vtx) const {
    return front_markers_[index_value(vtx) - offset_];
  }

  auto begin() const {
    return front_.begin();
  }
  auto end() const {
    return front_.end();
  }

private:
  void set(VertexIdx vtx, bool value) {
    front_markers_[index_value(vtx) - offset_] = value;
  }

  const VertexSize offset_;
  const VertexSize size_;
  thes::FixedBitset<sizeof(std::size_t)> front_markers_;
  thes::FlatSet<VertexIdx> front_{};
};

// Implements a sequential aggregation algorithm.
//
// Note that this implementation is not meant to be used directly with the AMG,
// but to be used as a helper for other aggregators, e.g. `SequentialAggregator`,
// `BlockedAggregator`, or `PartitionedAggregator`.
template<typename TAggMap, typename TGraph>
struct BaseAggregator {
  using AggregatesMap = TAggMap;
  using Graph = TGraph;
  using Size = thes::Intersection<typename TAggMap::Size, typename TGraph::Size>;
  using Vertex = Graph::Vertex;
  using FullVertex = Graph::FullVertex;
  using Edge = Graph::Edge;
  using EdgeProperties = Graph::EdgeProperties;
  using GraphSize = Graph::Size;
  using Index = AggregatesMap::Index;
  using VertexSize = AggregatesMap::VertexSize;
  using VertexIdx = AggregatesMap::VertexIdx;
  using Aggregate = AggregatesMap::Aggregate;
  using Front = AggregateFront<VertexIdx>;
  using VertexScrap = std::vector<VertexIdx>;
  using AggregateScrap = std::vector<Aggregate>;

  struct AggregateManager;

  // A helper class for managing aggregate identifiers.
  struct AggregateGenerator {
    friend struct AggregateManager;

    AggregateGenerator(const Graph& graph, AggregatesMap& aggregates)
        : graph_{graph}, aggregates_{aggregates} {
      aggregates_.push_aggregate();
    }

    void next_index() {
      aggregates_.push_aggregate();
      ++aggregate_counter_;
    }
    void finalize() {
      aggregates_.pop_aggregate();
    }
    [[nodiscard]] Index index() const {
      return Index{aggregate_counter_};
    }

    // The number of aggregates that have been created.
    [[nodiscard]] Size aggregate_num() const {
      return aggregate_counter_;
    }

  private:
    const Graph& graph_;
    // The aggregate mapping.
    AggregatesMap& aggregates_;
    // The number of aggregates that have already been created.
    Size aggregate_counter_{0};
  };

  // Some data structures to reduce memory allocations during the aggregation.
  struct AggregateHelpers {
    using AggregateSize = Aggregate::Size;

    Front front{};
    std::vector<AggregateSize> connected{};

    explicit AggregateHelpers(const Graph& graph)
        : front(graph.vertex_index_begin(), graph.vertex_index_end()) {}

    // Clear the aggregate helpers so that they can be re-used.
    void clear() {
      // Unmark all front vertices.
      front.clear();
      connected.clear();
    }
  };

  // A helper class for managing the current aggregate with its front and connected aggregates.
  struct AggregateManager {
    // The aggregate generator type.
    using Gen = AggregateGenerator;
    // The aggregate helpers type.
    using Helpers = AggregateHelpers;

    /**
     * @brief Constructs a new aggregate, which initially only contains `seed`.
     *
     * @param generator An aggregate generator.
     * @param seed The seed vertex of the aggregate, which is immediately added
     *  to the aggregate.
     * @param helpers The aggregate helpers.
     */
    AggregateManager(Gen& generator, const FullVertex& seed, Helpers& helpers)
        : seed_(seed), gen_{generator}, agg_{generator.index()}, helpers_{helpers} {
      assert(gen_.aggregates_.aggregate_size(agg_) == 0);
      add_help(seed);
    }
    AggregateManager(const AggregateManager&) = delete;
    AggregateManager& operator=(const AggregateManager&) = delete;
    AggregateManager(AggregateManager&&) = delete;
    AggregateManager& operator=(AggregateManager&&) = delete;

    ~AggregateManager() {
      helpers_.clear();
    }

    bool is_front(VertexIdx vtx) const {
      return helpers_.front.contains(vtx);
    }

    void add(const FullVertex& full_vtx) {
      helpers_.front.erase(full_vtx.index());
      add_help(full_vtx);
    }

    void add(std::vector<VertexIdx>& vertices) {
      [[maybe_unused]] const Size oldsize = vertex_num_;

      for (const VertexIdx vtx : vertices) {
        helpers_.front.erase(vtx);
        add_help(gen_.graph_.full_vertex_at(vtx));
      }

      assert(oldsize + vertices.size() == vertex_num_);
    }

    void merge_into(const Aggregate& aggregate) {
      assert(vertex_num_ == 1);
      gen_.aggregates_.move_to(seed_.index(), aggregate);
    }
    void move_to_isolated() {
      assert(vertex_num_ == 1);
      gen_.aggregates_.move_to_isolated(seed_.index());
    }

    void finalize() {
      gen_.next_index();
    }

    // Get the size of the aggregate.
    [[nodiscard]] Size size() const {
      assert(gen_.aggregates_.aggregate_size(agg_) == vertex_num_);
      return vertex_num_;
    }
    // Get the number of connections to other aggregates.
    [[nodiscard]] auto connect_size() const {
      return helpers_.connected.size();
    }

    // Get the identifier of the aggregate.
    [[nodiscard]] const Aggregate& aggregate() const {
      return agg_;
    }

    [[nodiscard]] const Front& front() const {
      return helpers_.front;
    }

    // Whether an aggregate is connected to this aggregate.
    [[nodiscard]] bool is_connected(const Aggregate& agg) const {
      for (auto idx : helpers_.connected) {
        if (Index{idx} == agg.index()) {
          return true;
        }
      }
      return false;
    }

  private:
    // A helper function that does everything necessary for adding a vertex to the aggregate,
    // except for removing the vertex from the front.
    THES_ALWAYS_INLINE void add_help(const FullVertex& vertex) {
      ++vertex_num_;
      gen_.aggregates_.store_aggregate(vertex.index(), agg_);
      assert(vertex_num_ == gen_.aggregates_.aggregate_size(agg_));

      vertex.iterate(
        thes::NoOp{},
        [&](auto edge) {
          const VertexIdx target = edge.head().index();
          const auto target_agg = gen_.aggregates_.aggregate_of(target);

          if (target_agg.is_aggregate()) {
            if (!is_connected(target_agg)) {
              helpers_.connected.push_back(index_value(target_agg.index()));
            }
          }
          if (target_agg.is_unaggregated() && !is_front(target)) {
            helpers_.front.insert(target);
          }
        },
        unvalued_tag, unordered_tag);
    }

    const FullVertex& seed_;
    // The aggregate generator.
    Gen& gen_;
    // The identifier of this aggregate.
    const Aggregate agg_;
    // The aggregate helpers.
    Helpers& helpers_;
    // The number of vertices in the aggregate.
    Size vertex_num_{0};
  };

  // Constructs an aggregator, which contains a reference to an aggregate mapping.
  explicit BaseAggregator(AggregatesMap& aggregates) : aggregates_{aggregates} {}

  BaseAggregator(const BaseAggregator&) = delete;
  BaseAggregator& operator=(const BaseAggregator&) = delete;
  BaseAggregator(BaseAggregator&&) noexcept = default;
  BaseAggregator& operator=(BaseAggregator&&) noexcept = default;

  ~BaseAggregator() = default;

  Aggregate aggregate_of(Vertex vertex) const {
    return aggregates_.aggregate_of(vertex.index());
  }
  void store_aggregate(Vertex vertex, Aggregate aggregate) const {
    return aggregates_.store_aggregate(vertex.index(), aggregate);
  }
  void store_isolated(Vertex vertex) const {
    return aggregates_.store_isolated(vertex.index());
  }

  // Returns the number of vertices stored in the aggregate mapping, including potential fillers
  // between 0 and the maximum vertex identifier.
  [[nodiscard]] Size vertex_num() const {
    return aggregates_.fine_num();
  }

  // A helper class storing information about an edge.
  struct EdgeInfo {
    const Edge& edge;
    EdgeProperties edge_props;
    // The aggregate of the edge head.
    const Aggregate head_aggregate;
    // The aggregate being built.
    const Aggregate current_aggregate;

    // Whether the edge head is in the aggregate being built.
    [[nodiscard]] bool head_in_aggregate() const {
      return head_aggregate == current_aggregate;
    }
  };

  // Count the number of neighbours that belong to the aggregate front.
  struct FrontNeighborCounter {
    explicit FrontNeighborCounter(const AggregateManager& agg_man) : agg_man_{agg_man} {}

    // Increment if the head of the edge is part of the aggregate front.
    void operator()(const EdgeInfo& ei) THES_ALWAYS_INLINE {
      counter_ += Size{agg_man_.is_front(ei.edge.head().index())};
    }
    [[nodiscard]] Size value(const Size /*neighbor_num*/) const THES_ALWAYS_INLINE {
      return counter_;
    }

  private:
    Size counter_{0};
    const AggregateManager& agg_man_;
  };

  // Count the two-way connected neighbours that belong to the aggregate being built.
  struct TwoWayConnectionCounter {
    TwoWayConnectionCounter() = default;

    // Increment if the edge head is in the aggregate being built and the edge is two-way strong.
    void operator()(const EdgeInfo& ei) THES_ALWAYS_INLINE {
      counter_ += Size{ei.head_in_aggregate() && ei.edge_props.is_two_way()};
    }
    [[nodiscard]] Size value(const Size /*neighbor_num*/) const THES_ALWAYS_INLINE {
      return counter_;
    }

  private:
    Size counter_{0};
  };

  // Count the one-way connected neighbours that belong to the aggregate being built.
  struct OneWayConnectionCounter {
    OneWayConnectionCounter() = default;

    // Increment if the edge head is in the aggregate being built and the edge is one-way strong.
    void operator()(const EdgeInfo& ei) THES_ALWAYS_INLINE {
      counter_ += Size{ei.head_in_aggregate() && ei.edge_props.is_one_way()};
    }
    [[nodiscard]] Size value(const Size /*neighbor_num*/) const THES_ALWAYS_INLINE {
      return counter_;
    }

  private:
    Size counter_{0};
  };

  // Count the dependent neighbours that belong to the aggregate being built.
  struct AnyWayConnectionCounter {
    AnyWayConnectionCounter() = default;

    // Increment if the edge head is in the aggregate being built and the edge is dependent.
    void operator()(const EdgeInfo& ei) THES_ALWAYS_INLINE {
      counter_ += Size{ei.head_in_aggregate() && ei.edge_props.depends()};
    }
    [[nodiscard]] Size value(const Size /*neighbor_num*/) const THES_ALWAYS_INLINE {
      return counter_;
    }

  private:
    Size counter_{0};
  };

  /**
   * @brief Compute the number of aggregated neighbours whose aggregate is connected
   *  to the current aggregate, which is divided by the number of neighbours
   *  in the `value()` function.
   *  This is dubbed “connectivity” in the literature on this method.
   *
   *  Adding a neighbour that fulfils the aforementioned condition does not connect
   *  a new aggregate to the current one, reducing the number of offdiagonal elements
   *  in the coarse matrix or, equivalently, the number of outgoing edges in the coarse graph.
   */
  struct ConnectivityCounter {
    explicit ConnectivityCounter(const AggregateManager& agg_man) : agg_man_{agg_man} {}

    // Increment if the head of the edge is aggregated and its aggregate is connected
    // to the aggregate being built.
    void operator()(const EdgeInfo& ei) THES_ALWAYS_INLINE {
      counter_ +=
        Size{ei.head_aggregate.is_aggregate() && agg_man_.is_connected(ei.head_aggregate)};
    }
    // The value of the counter divided by the number of neighbours.
    [[nodiscard]] double value(const Size neighbor_num) const THES_ALWAYS_INLINE {
      return double(counter_) / double(neighbor_num);
    }

  private:
    Size counter_{0};
    const AggregateManager& agg_man_;
  };

  struct AggregateNeighborsCounter {
    AggregateNeighborsCounter() = default;

    // Increments the counter if the head of the edge is in the aggregate being built.
    void operator()(const EdgeInfo& ei) THES_ALWAYS_INLINE {
      counter_ += Size{ei.head_in_aggregate()};
    }
    [[nodiscard]] Size value(const Size /*neighbor_num*/) const THES_ALWAYS_INLINE {
      return counter_;
    }

  private:
    Size counter_{0};
  };

  struct UnaggregatedNeighborsCounter {
    UnaggregatedNeighborsCounter() = default;

    // Increments the counter if the head of the edge is aggregated and its aggregate is connected
    // to the current aggregate.
    void operator()(const EdgeInfo& ei) THES_ALWAYS_INLINE {
      counter_ += Size{ei.head_aggregate.is_unaggregated()};
    }

    // The value of the counter.
    [[nodiscard]] Size value(const Size /*neighbor_num*/) const THES_ALWAYS_INLINE {
      return counter_;
    }

  private:
    Size counter_{0};
  };

  template<typename TParams>
  struct AggregationInstance {
    using Params = TParams;

    explicit AggregationInstance(AggregatesMap& aggregates, const Graph& graph, TParams&& params)
        : aggregates_(aggregates), graph_(graph), params_(std::forward<TParams>(params)) {}

    THES_ALWAYS_INLINE const AggregatesMap& aggregates_map() const {
      return aggregates_;
    }
    THES_ALWAYS_INLINE AggregatesMap& aggregates_map() {
      return aggregates_;
    }

    THES_ALWAYS_INLINE Aggregate aggregate_of(Vertex vertex) const {
      return aggregates_.aggregate_of(vertex.index());
    }
    THES_ALWAYS_INLINE void store_aggregate(Vertex vertex, Aggregate aggregate) const {
      return aggregates_.store_aggregate(vertex.index(), aggregate);
    }
    THES_ALWAYS_INLINE void store_isolated(Vertex vertex) const {
      return aggregates_.store_isolated(vertex.index());
    }

    // Applies each functor in args to the EdgeInfo created from each outgoing edge of vertex.
    // Returns a tuple of the values returned by the value function of each functor
    // after visiting each outgoing edge of vertex.
    THES_ALWAYS_INLINE auto apply_neighbors(const FullVertex& vertex,
                                            const AggregateManager& aggregate,
                                            auto&& counters) const {
      vertex.iterate(
        thes::NoOp{},
        [&, agg = aggregate.aggregate()](auto edge) THES_ALWAYS_INLINE {
          const EdgeInfo info(edge, graph_.get_edge_properties(edge), aggregate_of(edge.head()),
                              agg);
          counters | thes::star::for_each([&](auto& counter) THES_ALWAYS_INLINE { counter(info); });
        },
        unvalued_tag, unordered_tag);
      const auto adj_num = vertex.full_adjacent_num();
      return counters | thes::star::transform([&](const auto& counter) THES_ALWAYS_INLINE {
               return counter.value(adj_num);
             }) |
             thes::star::to_tuple;
    }

    // Searches for an aggregated neighbour of `vertex` the edge to whom is strong
    // and which has enough space for `vertex`.
    THES_ALWAYS_INLINE Aggregate find_merge_aggregate(const FullVertex& vertex,
                                                      const Params& params) const {
      return vertex.iterate(
        thes::NoOp<Aggregate>{},
        [&](auto&& edge) {
          if (!graph_.get_edge_properties(edge).is_strong()) {
            return Aggregate{};
          }
          if (const Aggregate agg = aggregate_of(edge.head());
              agg.is_aggregate() && aggregates_.aggregate_size(agg) < params.max_aggregate_size) {
            return agg;
          }
          return Aggregate{};
        },
        unvalued_tag, unordered_tag);
    }

    /**
     * @brief Grows `aggregate`, which initially only contains the non-isolated vertex `seed`,
     *  by repeatedly adding a selection of vertices from the aggregate front to the aggregate.
     *
     *  In more detail, this is done by taking the front of the aggregate as the initial set
     *  of candidates and filtering them as follows:
     *  - First, the candidates with the maximum number of two-way connections to the aggregate
     *    is chosen if any candidate has a two-way strong connection to the aggregate,
     *    otherwise the candidate with maximum number of one-way connections to the aggregate
     *    if any candidate has a one-way strong connection to the aggregate.
     *    If neither is the case, the algorithm is terminated.
     *  - Next, the candidates with maximal connectivity (`ConnectivityCounter`) are chosen.
     *  - Next, the candidates with the maximal number of neighbours in the front
     *     (`FrontNeighborCounter`) are chosen.
     *  - If adding all candidates would make `aggregate` have more elements than
     *    `criterion.maxAggregateSize()`, a sufficient number of elements is removed.
     *
     *  This is repeated until the size of the aggregate has reached `criterion.minAggregateSize()`
     *  or the diameter of the aggregate has reached `criterion.maxDistance()`.
     */
    void grow_aggregate(AggregateManager& aggregate) {
      while (aggregate.size() < params_.min_aggregate_size) {
        using Tup = thes::Tuple<Size, double, Size>;
        Tup max_tup{Size{0}, 0.0, Size{0}};
        bool has_cons2 = false;

        candidates_.clear();

        for (const VertexIdx vtx : aggregate.front()) {
          FullVertex full_vertex = graph_.full_vertex_at(vtx);

          // Only nonisolated nodes are considered
          if (graph_.is_isolated(full_vertex.vertex())) {
            continue;
          }

          const auto [con, nbs, cons2, cons1] = apply_neighbors(full_vertex, aggregate,
                                                                thes::Tuple{
                                                                  ConnectivityCounter{aggregate},
                                                                  FrontNeighborCounter{aggregate},
                                                                  TwoWayConnectionCounter{},
                                                                  OneWayConnectionCounter{},
                                                                });

          // Logic: If there is any two-way connection, only consider two-way vertices
          // Take the maximum of (cons, con, nbs)
          if (cons2 > 0 && !has_cons2) {
            // first two-way vertex → clear candidates
            has_cons2 = true;
            max_tup = Tup{cons2, con, nbs};
            candidates_.clear();
            candidates_.push_back(vtx);
          } else if (Tup tup{has_cons2 ? cons2 : cons1, con, nbs}; thes::star::get_at<0>(tup) > 0) {
            // stayed in the same mode
            std::partial_ordering cmp = tup <=> max_tup;
            if (cmp == std::partial_ordering::greater) {
              max_tup = tup;
              candidates_.clear();
              candidates_.push_back(vtx);
            } else if (cmp == std::partial_ordering::equivalent) {
              candidates_.push_back(vtx);
            }
          }
        }

        if (candidates_.empty()) {
          break; // No more candidates found
        }

        candidates_.resize(
          std::min<std::size_t>(candidates_.size(), params_.max_aggregate_size - aggregate.size()));
        aggregate.add(candidates_);
      }
    }

    /**
     * @brief Rounds the aggregate by repeatedly adding a selection of vertices
     *  from the aggregate front to the aggregate.
     *
     *  In more detail, this means taking the non-isolated vertices in the front of `aggregate`
     *  that have one- or two-way strong connections to `aggregate` and which have more connections
     *  to `aggregate` than unaggregated vertices.
     *  If adding all these candidates would make `aggregate` have more elements than
     *  `criterion.maxAggregateSize()`, a sufficient number of elements is removed
     *  from the candidate set. The remaining candidates are added to the aggregate.
     *
     *  This is repeated until the aggregate has reached a size of `criterion.maxAggregateSize()`
     *  or the candidate set is empty.
     */
    void round_aggregate(AggregateManager& aggregate) {
      while (aggregate.size() < params_.max_aggregate_size) {
        candidates_.clear();

        for (const VertexIdx vtx : aggregate.front()) {
          FullVertex full_vertex = graph_.full_vertex_at(vtx);

          if (graph_.is_isolated(full_vertex.vertex())) {
            continue; // No isolated nodes here
          }

          const auto [cons, unaggregated, in_aggregate] =
            apply_neighbors(full_vertex, aggregate,
                            thes::Tuple{
                              AnyWayConnectionCounter{},
                              UnaggregatedNeighborsCounter{},
                              AggregateNeighborsCounter{},
                            });
          if (cons == 0 || unaggregated >= in_aggregate) {
            continue;
          }

          candidates_.push_back(vtx);
          break;
        }

        if (candidates_.empty()) {
          // no more candidates found.
          break;
        }

        candidates_.resize(
          std::min<std::size_t>(candidates_.size(), params_.max_aggregate_size - aggregate.size()));
        aggregate.add(candidates_);
      }
    }

    void handle_vertex(const FullVertex& seed, bool is_aggressive) {
      const Vertex vertex = seed.vertex();
      if (aggregate_of(vertex).is_aggregated()) {
        return;
      }

      if (graph_.is_isolated(vertex)) {
        store_isolated(vertex);
        ++iso_aggregates_;
        return;
      }

      AggregateManager aggregate{generator_, seed, helpers_};
      grow_aggregate(aggregate);
      round_aggregate(aggregate);

      // try to merge aggregates consisting of only one nonisolated vertex with other aggregates
      if (aggregate.size() == 1 && params_.max_aggregate_size > 1) {
        const auto merge_aggregate = find_merge_aggregate(seed, params_);

        if (merge_aggregate.is_aggregated()) {
          // assign vertex to the neighbouring cluster
          aggregate.merge_into(merge_aggregate);
          ++merged_aggregates_;
          return;
        }

        if (is_aggressive) {
          aggregate.move_to_isolated();
          ++iso_aggregates_;
          return;
        }

        ++one_aggregates_;
      } else {
        ++multi_aggregates_;
      }

      aggregate.finalize();
    }

    AggregationStats finalize() {
      generator_.finalize();
      const Size aggregate_num = generator_.aggregate_num();
      assert(aggregate_num == multi_aggregates_ + one_aggregates_);

      return {
        .aggregate_num = aggregate_num,
        .multi_vertex_aggregate_num = multi_aggregates_,
        .single_vertex_aggregate_num = one_aggregates_,
        .merged_aggregate_num = merged_aggregates_,
        .isolated_aggregate_num = iso_aggregates_,
      };
    }

  private:
    AggregatesMap& aggregates_;
    const Graph& graph_;
    TParams params_;

    Size multi_aggregates_{0};
    Size one_aggregates_{0};
    Size merged_aggregates_{0};
    Size iso_aggregates_{0};

    AggregateGenerator generator_{graph_, aggregates_};

    VertexScrap candidates_{};
    AggregateHelpers helpers_{graph_};
  };

private:
  AggregatesMap& aggregates_;
};
} // namespace lineal

#endif // INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_AGGREGATION_BASE_HPP
