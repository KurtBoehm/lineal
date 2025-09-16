// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_AGGREGATION_INDEPENDENT_HPP
#define INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_AGGREGATION_INDEPENDENT_HPP

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "thesauros/io.hpp"
#include "thesauros/macropolis.hpp"
#include "thesauros/static-ranges.hpp"
#include "thesauros/utility.hpp"

#include "lineal/base.hpp"
#include "lineal/component-wise.hpp"
#include "lineal/linear-solver/multigrid/aggregation/base.hpp"
#include "lineal/linear-solver/multigrid/aggregation/params.hpp"
#include "lineal/linear-solver/multigrid/aggregation/stats.hpp"
#include "lineal/linear-solver/multigrid/graph/propertied-subgraph.hpp"
#include "lineal/linear-solver/multigrid/types.hpp"
#include "lineal/vectorization.hpp"

namespace lineal::amg {
namespace detail {
template<SharedGraph TGraph>
inline std::filesystem::path level_path(std::size_t level_idx, const TGraph& graph,
                                        std::filesystem::path path) {
  if constexpr (TGraph::is_shared) {
    path += std::to_string(level_idx) + ".aggs";
  } else {
    path += std::to_string(level_idx) + "-" + std::to_string(graph.distributed_info().comm_rank()) +
            ".aggs";
  }
  return path;
}
} // namespace detail

template<typename TAggMap, typename TGraph, typename TParams, Env TEnv>
struct IndependentAggregationSink {
  using AggregatesMap = TAggMap;
  using Graph = TGraph;
  using VertexIdx = Graph::VertexIdx;

  using AggMapBuilder = AggregatesMap::Builder;
  using AggExecInstance = AggMapBuilder::ExecInstance;
  using Size = AggregatesMap::Size;

  template<typename TThreadInfo>
  using VerticesSubGraph =
    PropertiedSubgraph<const Graph&, typename std::decay_t<TThreadInfo>::Indices>;

  using Params = std::decay_t<TParams>;
  static constexpr bool is_shared = Graph::is_shared;
  static constexpr auto exec_constraints =
    thes::star::joined(lineal::exec_constraints<Graph>,
                       thes::Tuple{ForwardIterConstraint{}, ThreadSeqIterConstraint{}}) |
    thes::star::to_tuple;

  template<typename TThreadInfo>
  struct ThreadSubGraphData {
    using SubGraph = VerticesSubGraph<TThreadInfo>;

    ThreadSubGraphData(TThreadInfo&& tinfo, const Graph& graph)
        : thread_info{std::forward<TThreadInfo>(tinfo)},
          subgraph(graph, to_index_range(thread_info.indices(), thes::type_tag<VertexIdx>)) {}

    TThreadInfo thread_info;
    SubGraph subgraph;
  };

private:
  struct SinkConf {
    using Work = void;
    using Value = void;
    static constexpr bool supports_const_access = false;
    static constexpr bool supports_mutable_access = true;
  };

public:
  template<typename TThreadInfo>
  struct ThreadInstance
      : public ThreadSubGraphData<TThreadInfo>,
        public facades::ComponentWiseOp<ThreadInstance<TThreadInfo>, SinkConf,
                                        typename VerticesSubGraph<TThreadInfo>::FullView> {
    using ThreadInfo = std::decay_t<TThreadInfo>;
    using Vertices = ThreadInfo::Indices;
    using SubGraph = VerticesSubGraph<ThreadInfo>;
    using VertexIter = SubGraph::const_iterator;
    using FullVertex = SubGraph::FullVertex;
    using FullView = SubGraph::FullView;

    using Parent = facades::ComponentWiseOp<ThreadInstance, SinkConf, FullView>;

    using Data = ThreadSubGraphData<TThreadInfo>;
    using AggregatesThreadInstance = AggExecInstance::template ThreadInstance<Vertices>;
    using AggregationInstance =
      BaseAggregator<AggregatesThreadInstance,
                     SubGraph>::template AggregationInstance<const Params&>;

    ThreadInstance(const Graph& graph, const Params& params,
                   AggExecInstance& multithread_aggregates, TThreadInfo&& thread_info,
                   bool is_aggressive, AggregationStats& stats)
        : Data(std::forward<TThreadInfo>(thread_info), graph), Parent(FullView(this->subgraph)),
          thread_instance_(multithread_aggregates.thread_instance(
            this->thread_info.index(), this->thread_info.indices(), params.max_aggregate_size)),
          aggregation_instance_(thread_instance_, this->subgraph, params),
          is_aggressive_(is_aggressive), stats_(stats) {}

    ThreadInstance(const ThreadInstance&) = delete;
    ThreadInstance& operator=(const ThreadInstance&) = delete;
    ThreadInstance(ThreadInstance&&) noexcept = default;
    ThreadInstance& operator=(ThreadInstance&&) noexcept = default;

    THES_ALWAYS_INLINE void compute_iter(grex::OptValuedScalarTag<Size> auto /*tag*/,
                                         const auto& /*children*/, auto vtx_it) {
      aggregation_instance_.handle_vertex(*vtx_it, is_aggressive_);
    }

    ~ThreadInstance() {
      AggregationStats stats = aggregation_instance_.finalize();
      stats_ += stats;
    }

  private:
    AggregatesThreadInstance thread_instance_;
    AggregationInstance aggregation_instance_;
    bool is_aggressive_;
    AggregationStats& stats_;
  };

  struct ExecInstance {
    template<typename TIdxs>
    using ThreadInstance = IndependentAggregationSink::ThreadInstance<TIdxs>;

    ExecInstance(std::size_t thread_num, AggMapBuilder& aggregates_builder, const Graph& graph,
                 const Params& params, bool is_aggressive, const TEnv& env)
        : graph_(graph), params_(params), env_(env), is_aggressive_(is_aggressive),
          multithread_aggregates_(aggregates_builder.exec_instance(thread_num)) {}
    ExecInstance(const ExecInstance&) = delete;
    ExecInstance(ExecInstance&&) = delete;
    ExecInstance& operator=(const ExecInstance&) = delete;
    ExecInstance& operator=(ExecInstance&&) = delete;
    ~ExecInstance() {
      env_.log("aggregation_stats", stats_);
    }

    template<typename TThreadInfo>
    ThreadInstance<TThreadInfo> thread_instance(TThreadInfo&& thread_info,
                                                grex::OptValuedScalarTag<Size> auto /*tag*/) {
      return {
        graph_,         params_, multithread_aggregates_, std::forward<TThreadInfo>(thread_info),
        is_aggressive_, stats_,
      };
    }

  private:
    const Graph& graph_;
    const Params& params_;
    const TEnv& env_;
    bool is_aggressive_;
    AggExecInstance multithread_aggregates_;
    AggregationStats stats_{};
  };

  explicit IndependentAggregationSink(const Graph& graph, AggMapBuilder& aggregates_builder,
                                      TParams&& params, bool is_aggressive, const TEnv& env)
      : graph_(graph), agg_builder_(aggregates_builder), env_(env),
        params_(std::forward<TParams>(params)), is_aggressive_(is_aggressive) {}

  [[nodiscard]] Size size() const {
    return graph_.vertex_num();
  }

  [[nodiscard]] thes::Optional<std::size_t> thread_num(std::size_t expo_size) const {
    return thes::Optional{params_.min_per_thread}.transform(
      [&](std::size_t m) { return std::clamp<std::size_t>(size() / m, 1, expo_size); });
  }

  ExecInstance exec_instance(std::size_t thread_num) {
    return ExecInstance(thread_num, agg_builder_, graph_, params_, is_aggressive_, env_);
  }

  static AggregatesMap build_map(const TGraph& graph, TParams&& params, bool is_aggressive,
                                 const Env auto& env) {
    AggMapBuilder builder(graph.vertex_index_end());

    {
      auto envi = env.add_object("independent_aggregation");

      IndependentAggregationSink sink{graph, builder, std::forward<TParams>(params), is_aggressive,
                                      envi};
      decltype(auto) expo = env.execution_policy();
      builder.initialize(expo, sink.thread_num(expo.thread_num()));

      expo.execute(sink);
    }

    return builder.build();
  }

private:
  const Graph& graph_;
  AggMapBuilder& agg_builder_;
  const TEnv& env_;
  TParams params_;
  bool is_aggressive_;
};

template<typename TAggMap>
struct HomogeneousAggregator {
  using AggregatesMap = TAggMap;
  using Size = AggregatesMap::Size;
  using Params = AggregationParams<Size>;

  explicit HomogeneousAggregator(Params params) : params_(std::move(params)) {}

  template<SharedGraph TGraph, Env TEnv>
  AggregatesMap aggregate(const TGraph& graph, std::size_t /*level_idx*/, bool is_aggressive,
                          const TEnv& env) const {
    using Sink = IndependentAggregationSink<TAggMap, TGraph, const Params&, TEnv>;
    return Sink::build_map(graph, params_, is_aggressive, env);
  }

private:
  Params params_;
};

template<typename TAggMap>
struct RefinedAggregator {
  using AggregatesMap = TAggMap;
  using Size = AggregatesMap::Size;
  using LevelParams = AggregationParams<Size>;

  struct Params {
    using Range = ClosedRange<Size>;

    LevelParams on_level(std::size_t idx) const {
      const auto range = (idx < aggregate_size_ranges.size()) ? aggregate_size_ranges[idx]
                                                              : aggregate_size_ranges.back();
      return LevelParams{
        .min_aggregate_size = range.min,
        .max_aggregate_size = range.max,
        .min_per_thread = min_per_thread,
      };
    }

    std::vector<ClosedRange<Size>> aggregate_size_ranges{{.min = 6, .max = 9}};
    std::optional<Size> min_per_thread{};
    std::optional<std::filesystem::path> base_agg_map_path{};
  };

  explicit RefinedAggregator(Params params) : params_(std::move(params)) {}

  template<SharedGraph TGraph, Env TEnv>
  AggregatesMap aggregate(const TGraph& graph, std::size_t level_idx, bool is_aggressive,
                          const TEnv& env) const {
    using Sink = IndependentAggregationSink<TAggMap, TGraph, LevelParams, TEnv>;
    auto agg_map = Sink::build_map(graph, params_.on_level(level_idx), is_aggressive, env);
    if (params_.base_agg_map_path.has_value()) {
      thes::FileWriter writer{detail::level_path(level_idx, graph, *params_.base_agg_map_path)};
      agg_map.to_file(writer);
    }
    return agg_map;
  }

private:
  Params params_;
};

// WARNING This simply tries to load the files at the given paths;
// if these do not exist, have incorrect dimensions, or do not match a given matrix,
// this approach will not work.
template<typename TAggMap>
struct FileAggregator {
  using AggregatesMap = TAggMap;
  using Size = AggregatesMap::Size;

  struct Params {
    std::filesystem::path base_agg_map_path = thes::init_required;
  };

  explicit FileAggregator(Params params) : params_(std::move(params)) {}

  template<SharedGraph TGraph>
  AggregatesMap aggregate(const TGraph& graph, std::size_t level_idx, bool /*is_aggressive*/,
                          const Env auto& /*env*/) const {
    thes::FileReader reader{detail::level_path(level_idx, graph, params_.base_agg_map_path)};
    return AggregatesMap::from_file(reader);
  }

private:
  Params params_;
};
} // namespace lineal::amg

#endif // INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_AGGREGATION_INDEPENDENT_HPP
