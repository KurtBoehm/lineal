// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_TRANSPORT_VECTOR_SHARED_HPP
#define INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_TRANSPORT_VECTOR_SHARED_HPP

#include <type_traits>
#include <utility>

#include "thesauros/macropolis.hpp"

#include "lineal/base.hpp"
#include "lineal/component-wise/facade.hpp"
#include "lineal/linear-solver/multigrid/transport/info/simple.hpp"
#include "lineal/parallel/def/index.hpp"
#include "lineal/parallel/index/index.hpp"
#include "lineal/vectorization.hpp"

namespace lineal {
// coarsening

template<typename TReal, AnyVector TVec, typename TAggMap>
struct SharedVectorCoarsener {
  using Vec = std::decay_t<TVec>;
  using VecReal = Vec::Value;
  using AggregatesMap = std::decay_t<TAggMap>;

  using Size = AggregatesMap::Size;
  using Index = AggregatesMap::Index;
  using Aggregate = AggregatesMap::Aggregate;

  struct Conf {
    using Value = TReal;
    using Size = SharedVectorCoarsener::Size;
    using IndexTag = OwnIndexTag;

    static constexpr bool has_local_view = false;
  };

  struct ExprBase {
    ExprBase(TVec&& fine_vector, TAggMap&& aggregates_map)
        : fine_vector_(std::forward<TVec>(fine_vector)),
          aggregates_map_(std::forward<TAggMap>(aggregates_map)) {}

    THES_ALWAYS_INLINE constexpr TReal compute_impl(grex::ScalarTag /*tag*/, Size index) const {
      TReal sum = 0;
      for (decltype(auto) vertex :
           aggregates_map_.coarse_row_to_fine_rows(Aggregate{Index{index}})) {
        sum += fine_vector_[vertex];
      }
      return sum;
    }

    THES_ALWAYS_INLINE constexpr const TVec& fine_vector() const {
      return fine_vector_;
    }
    THES_ALWAYS_INLINE constexpr const TAggMap& aggregates_map() const {
      return aggregates_map_;
    }

  private:
    TVec fine_vector_;
    TAggMap aggregates_map_;
  };

  struct SharedExpr : public ExprBase, public facades::SharedNullaryCwOp<SharedExpr, Conf> {
    using Facade = facades::SharedNullaryCwOp<SharedExpr, Conf>;
    static constexpr auto exec_constraints = lineal::exec_constraints<Vec>;

    SharedExpr(TVec&& fine_vector, TAggMap&& aggregates_map)
        : ExprBase(std::forward<TVec>(fine_vector), std::forward<TAggMap>(aggregates_map)),
          Facade(this->aggregates_map().coarse_row_num()) {}
  };
};

template<typename TReal, SharedVector TFineVec, typename TAggMap>
static constexpr auto coarsen_vector(TFineVec&& fine_vector, TAggMap&& aggregates_map) {
  using Expr = SharedVectorCoarsener<TReal, TFineVec, TAggMap>::SharedExpr;
  return Expr{std::forward<TFineVec>(fine_vector), std::forward<TAggMap>(aggregates_map)};
}
template<typename TReal, SharedVector TCoarseVec, SharedVector TFineVec, typename TSizeByte,
         typename TByteAlloc>
static constexpr void coarsen_vector(TCoarseVec&& coarse_vector, TFineVec&& fine_vector,
                                     const SimpleTransportInfo<TSizeByte, TByteAlloc>& trans,
                                     const auto& env) {
  assign(std::forward<TCoarseVec>(coarse_vector),
         coarsen_vector<TReal>(std::forward<TFineVec>(fine_vector), trans.aggregate_map()),
         env.execution_policy());
}

// refinement

template<AnyVector TVec, typename TAggMap>
struct SharedVectorRefiner {
  using Vec = std::decay_t<TVec>;
  using VecReal = Vec::Value;
  using GlobalSize = Vec::GlobalSize;
  using AggregatesMap = std::decay_t<TAggMap>;

  using Size = AggregatesMap::Size;
  using Aggregate = AggregatesMap::Aggregate;

  struct Conf {
    using Value = VecReal;
    using Size = SharedVectorRefiner::Size;
    using IndexTag = OwnIndexTag;

    static constexpr bool has_local_view = false;
  };

  template<typename TDerived>
  struct ExprBase {
    ExprBase(TVec&& coarse_vector, TAggMap&& aggregates_map)
        : coarse_vector_(std::forward<TVec>(coarse_vector)),
          aggregates_map_(std::forward<TAggMap>(aggregates_map)) {}

    THES_ALWAYS_INLINE constexpr auto compute_impl(grex::ScalarTag /*tag*/,
                                                   AnyTypedIndex<Size, GlobalSize> auto idx) const {
      const auto aggregate =
        aggregates_map_[index_value(idx, own_index_tag, derived().distributed_info_storage())];
      return aggregate.is_aggregate() ? coarse_vector_[aggregate.index()] : 0;
    }
    THES_ALWAYS_INLINE constexpr auto compute_impl(grex::AnyVectorTag auto tag,
                                                   AnyTypedIndex<Size, GlobalSize> auto idx) const {
      const auto agg = aggregates_map_.load(
        index_value(idx, own_index_tag, derived().distributed_info_storage()), tag);
      const auto is_agg = grex::convert_safe<VecReal>(agg.is_aggregate());
      return grex::mask_gather(std::as_const(coarse_vector_).span(), is_agg, *agg, tag);
    }

    THES_ALWAYS_INLINE constexpr const TVec& coarse_vector() const {
      return coarse_vector_;
    }
    THES_ALWAYS_INLINE constexpr const TAggMap& aggregates_map() const {
      return aggregates_map_;
    }

  private:
    const TDerived& derived() const {
      return static_cast<const TDerived&>(*this);
    }

    TVec coarse_vector_;
    TAggMap aggregates_map_;
  };

  struct SharedExpr : public ExprBase<SharedExpr>,
                      public facades::SharedNullaryCwOp<SharedExpr, Conf> {
    using Facade = facades::SharedNullaryCwOp<SharedExpr, Conf>;
    static constexpr auto exec_constraints = lineal::exec_constraints<Vec>;

    SharedExpr(TVec&& coarse_vector, TAggMap&& aggregates_map)
        : ExprBase<SharedExpr>(std::forward<TVec>(coarse_vector),
                               std::forward<TAggMap>(aggregates_map)),
          Facade(this->aggregates_map().fine_row_num()) {}
  };
};

template<SharedVector TCoarseVec, typename TAggMap>
static constexpr auto refine_vector(TCoarseVec&& coarse_vector, TAggMap&& aggregates_map) {
  using Expr = SharedVectorRefiner<TCoarseVec, TAggMap>::SharedExpr;
  return Expr{std::forward<TCoarseVec>(coarse_vector), std::forward<TAggMap>(aggregates_map)};
}
template<SharedVector TCoarseVec, typename TSizeByte, typename TByteAlloc>
static constexpr auto refine_vector(TCoarseVec&& coarse_vector,
                                    const SimpleTransportInfo<TSizeByte, TByteAlloc>& trans,
                                    const auto& /*env*/) {
  return refine_vector(std::forward<TCoarseVec>(coarse_vector), trans.aggregate_map());
}
} // namespace lineal

#endif // INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_TRANSPORT_VECTOR_SHARED_HPP
