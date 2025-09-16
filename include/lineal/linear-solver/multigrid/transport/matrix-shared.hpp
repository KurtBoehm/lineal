// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_TRANSPORT_MATRIX_SHARED_HPP
#define INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_TRANSPORT_MATRIX_SHARED_HPP

#include <cassert>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "thesauros/macropolis.hpp"
#include "thesauros/static-ranges.hpp"
#include "thesauros/utility.hpp"

#include "lineal/base.hpp"
#include "lineal/component-wise.hpp"
#include "lineal/linear-solver/multigrid/transport/info/simple.hpp"
#include "lineal/parallel/def/index.hpp"
#include "lineal/vectorization.hpp"

namespace lineal {
template<typename TReal, SharedMatrix TFineMat, SharedMatrix TCoarseMat, typename TAggMap>
struct LocalMatrixCoarsener {
  using Real = TReal;
  using FineMatrix = std::decay_t<TFineMat>;
  using CoarseMatrix = TCoarseMat;
  using AggregatesMap = TAggMap;

  using FineExtRowIdx = FineMatrix::ExtRowIdx;

  using CoarseValue = CoarseMatrix::Value;
  using Size = CoarseMatrix::Size;
  using DistributedInfo = CoarseMatrix::DistributedInfo;
  using DistributedInfoStorage = CoarseMatrix::DistributedInfoStorage;
  using Aggregate = AggregatesMap::Aggregate;
  using RowIdx = AggregatesMap::Index;

  using Planner = CoarseMatrix::template MultithreadPlanner<NonUniqueTag, OwnIndexTag>;
  using PlannerThreadInstance = Planner::ThreadInstance;
  using Builder = CoarseMatrix::template MultithreadBuilder<Real, NonUniqueTag, OwnIndexTag>;
  using BuilderThreadInstance = Builder::ThreadInstance;

  static constexpr bool is_shared = std::is_void_v<DistributedInfo>;

  struct SinkConf {
    using Value = void;
    using Size = LocalMatrixCoarsener::Size;
    using IndexTag = OwnIndexTag;

    static constexpr bool supports_const_access = false;
    static constexpr bool supports_mutable_access = true;
  };

  struct Phase1Sink {
    using Size = LocalMatrixCoarsener::Size;
    static constexpr bool is_shared = LocalMatrixCoarsener::is_shared;
    static constexpr auto exec_constraints =
      thes::star::joined(lineal::exec_constraints<FineMatrix>,
                         thes::Tuple{ForwardIterConstraint{}, ThreadSeqIterConstraint{}}) |
      thes::star::to_tuple;

    struct ExecInstance {
      struct ThreadInstance : public facades::SharedNullaryCwOp<ThreadInstance, SinkConf> {
        using Parent = facades::SharedNullaryCwOp<ThreadInstance, SinkConf>;

        ThreadInstance(std::size_t thread_index, Planner& planner, const FineMatrix& fine_matrix,
                       const AggregatesMap& aggregates_map)
            : Parent(aggregates_map.coarse_row_num()), thread_index_(thread_index),
              thread_instance_(planner.thread_instance(thread_index)), fine_matrix_(fine_matrix),
              aggregates_map_(aggregates_map) {}

        ThreadInstance(ThreadInstance&&) = delete;
        ThreadInstance(const ThreadInstance&) = delete;
        ThreadInstance& operator=(ThreadInstance&&) = delete;
        ThreadInstance& operator=(const ThreadInstance&) = delete;

        ~ThreadInstance() {
          thread_instance_.finalize();
        }

        constexpr auto compute_impl(grex::ScalarTag /*tag*/, Size index) THES_ALWAYS_INLINE {
          for (decltype(auto) vertex :
               aggregates_map_.coarse_row_to_fine_rows(Aggregate{RowIdx{index}})) {
            fine_matrix_[vertex].iterate(
              [&](FineExtRowIdx col) {
                auto&& aggregate = aggregates_map_[col];
                assert(aggregate.is_aggregated());
                if (aggregate.is_aggregate()) {
                  thread_instance_.add_column(col_idx_of(aggregate));
                }
              },
              unvalued_tag, unordered_tag);
          }
          ++thread_instance_;
        }

      private:
        std::size_t thread_index_;
        PlannerThreadInstance thread_instance_;
        const FineMatrix& fine_matrix_;
        const AggregatesMap& aggregates_map_;
      };

      ExecInstance(const FineMatrix& fine_matrix, const AggregatesMap& aggregates_map,
                   Planner& planner)
          : fine_matrix_(fine_matrix), aggregates_map_(aggregates_map), planner_(planner) {}

      ThreadInstance thread_instance(const auto& thread_info, grex::ScalarTag /*tag*/) {
        return {thread_info.index(), planner_, fine_matrix_, aggregates_map_};
      }

    private:
      const FineMatrix& fine_matrix_;
      const AggregatesMap& aggregates_map_;
      Planner& planner_;
    };

    explicit constexpr Phase1Sink(const FineMatrix& fine_matrix,
                                  const AggregatesMap& aggregates_map, Planner& planner)
        : fine_matrix_(fine_matrix), aggregates_map_(aggregates_map), planner_(planner) {}

    [[nodiscard]] Size size() const {
      return aggregates_map_.coarse_row_num();
    }

    constexpr ExecInstance exec_instance(std::size_t thread_num) {
      planner_.initialize(thread_num, fine_matrix_.is_symmetric());
      return ExecInstance(fine_matrix_, aggregates_map_, planner_);
    }

  private:
    const FineMatrix& fine_matrix_;
    const AggregatesMap& aggregates_map_;
    Planner& planner_;
  };

  struct Phase2Sink {
    using Size = LocalMatrixCoarsener::Size;
    static constexpr bool is_shared = LocalMatrixCoarsener::is_shared;
    static constexpr thes::Tuple exec_constraints{
      ForwardIterConstraint{},
      ThreadSeqIterConstraint{},
    };

    struct ExecInstance {
      struct ThreadInstance : public facades::SharedNullaryCwOp<ThreadInstance, SinkConf> {
        using Parent = facades::SharedNullaryCwOp<ThreadInstance, SinkConf>;

        ThreadInstance(std::size_t thread_index, Builder& builder, const FineMatrix& fine_matrix,
                       const AggregatesMap& aggregates_map)
            : Parent(aggregates_map.coarse_row_num()), thread_index_(thread_index),
              thread_instance_(builder.thread_instance(builder.row_offset(thread_index),
                                                       builder.non_zero_offset(thread_index))),
              fine_matrix_(fine_matrix), aggregates_map_(aggregates_map) {}

        ThreadInstance(ThreadInstance&&) = delete;
        ThreadInstance(const ThreadInstance&) = delete;
        ThreadInstance& operator=(ThreadInstance&&) = delete;
        ThreadInstance& operator=(const ThreadInstance&) = delete;

        ~ThreadInstance() {
          thread_instance_.finalize();
        }

        THES_ALWAYS_INLINE constexpr auto compute_impl(grex::ScalarTag /*tag*/, Size index) {
          for (decltype(auto) vertex :
               aggregates_map_.coarse_row_to_fine_rows(Aggregate{RowIdx{index}})) {
            fine_matrix_[vertex].iterate(
              [&](auto col, auto&& val) {
                auto&& aggregate = aggregates_map_[col];
                assert(aggregate.is_aggregated());
                if (aggregate.is_aggregate()) {
                  thread_instance_.insert(col_idx_of(aggregate), static_cast<CoarseValue>(val));
                }
              },
              valued_tag, unordered_tag);
          }
          ++thread_instance_;
        }

      private:
        std::size_t thread_index_;
        BuilderThreadInstance thread_instance_;
        const FineMatrix& fine_matrix_;
        const AggregatesMap& aggregates_map_;
      };

      ExecInstance(const FineMatrix& fine_matrix, const AggregatesMap& aggregates_map,
                   Builder& builder)
          : fine_matrix_(fine_matrix), aggregates_map_(aggregates_map), builder_(builder) {}

      ThreadInstance thread_instance(const auto& thread_info, grex::ScalarTag /*tag*/) {
        return {thread_info.index(), builder_, fine_matrix_, aggregates_map_};
      }

    private:
      const FineMatrix& fine_matrix_;
      const AggregatesMap& aggregates_map_;
      Builder& builder_;
    };

    explicit constexpr Phase2Sink(const FineMatrix& fine_matrix,
                                  const AggregatesMap& aggregates_map, Planner& planner,
                                  Builder& builder, DistributedInfoStorage&& coarse_dist_info)
        : fine_matrix_(fine_matrix), aggregates_map_(aggregates_map), planner_(planner),
          builder_(builder),
          coarse_dist_info_(std::forward<DistributedInfoStorage>(coarse_dist_info)) {}

    [[nodiscard]] Size size() const {
      return aggregates_map_.coarse_row_num();
    }

    constexpr ExecInstance exec_instance(std::size_t /*thread_num*/) {
      builder_.initialize(std::move(planner_),
                          std::forward<DistributedInfoStorage>(coarse_dist_info_));
      return ExecInstance(fine_matrix_, aggregates_map_, builder_);
    }

  private:
    const FineMatrix& fine_matrix_;
    const AggregatesMap& aggregates_map_;
    Planner& planner_;
    Builder& builder_;
    [[no_unique_address]] DistributedInfoStorage coarse_dist_info_;
  };

  static TCoarseMat coarsen(const TFineMat& fine_matrix, const TAggMap& aggregates_map,
                            DistributedInfoStorage&& dist_info, const auto& env)
  requires(!is_shared)
  {
    return coarsen_impl([&](Builder&& builder) { return std::move(builder).build(); }, fine_matrix,
                        aggregates_map, std::forward<DistributedInfoStorage>(dist_info), env);
  }

  static TCoarseMat coarsen(const TFineMat& fine_matrix, const TAggMap& aggregates_map,
                            const auto& env)
  requires(is_shared)
  {
    return coarsen_impl([](Builder&& builder) { return std::move(builder).build(); }, fine_matrix,
                        aggregates_map, thes::Empty{}, env);
  }

private:
  static auto col_idx_of(auto agg) {
    if constexpr (is_shared) {
      return index_value(agg.index());
    } else {
      return agg.index();
    }
  }

  static TCoarseMat coarsen_impl(auto build, const TFineMat& fine_matrix,
                                 const TAggMap& aggregates_map, DistributedInfoStorage&& dist_info,
                                 const auto& env) {
    const auto& expo = env.execution_policy();

    Planner planner{};
    {
      Phase1Sink sink{fine_matrix, aggregates_map, planner};
      expo.execute(sink);
    }

    Builder builder{};
    {
      Phase2Sink sink{fine_matrix, aggregates_map, planner, builder,
                      std::forward<DistributedInfoStorage>(dist_info)};
      expo.execute(sink);
    }

    return build(std::move(builder));
  }
};

template<typename TReal, SharedMatrix TCoarseMat, SharedMatrix TFineMat, typename TAggMap>
requires std::is_void_v<typename TCoarseMat::DistributedInfo>
TCoarseMat coarsen_matrix(const TFineMat& fine_matrix, const TAggMap& aggregates_map,
                          const auto& env) {
  using Coarsener = LocalMatrixCoarsener<TReal, TFineMat, TCoarseMat, TAggMap>;
  return Coarsener::coarsen(fine_matrix, aggregates_map, env);
}
template<typename TReal, SharedMatrix TCoarseMat, SharedMatrix TFineMat, typename TByteAlloc>
requires std::is_void_v<typename TCoarseMat::DistributedInfo>
TCoarseMat coarsen_matrix(const TFineMat& fine_matrix,
                          const SimpleTransportInfo<typename TFineMat::SizeByte, TByteAlloc>& trans,
                          const auto& env) {
  return coarsen_matrix<TReal, TCoarseMat>(fine_matrix, trans.aggregate_map(), env);
}
} // namespace lineal

#endif // INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_TRANSPORT_MATRIX_SHARED_HPP
