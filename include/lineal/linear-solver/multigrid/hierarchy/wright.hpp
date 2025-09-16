// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_HIERARCHY_WRIGHT_HPP
#define INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_HIERARCHY_WRIGHT_HPP

#include <cstddef>
#include <exception>
#include <tuple>
#include <type_traits>
#include <utility>

#include "thesauros/macropolis.hpp"
#include "thesauros/static-ranges.hpp"
#include "thesauros/types.hpp"

#include "lineal/base.hpp"
#include "lineal/linear-solver/multigrid/hierarchy/hierarchy.hpp"
#include "lineal/linear-solver/multigrid/transport.hpp"
#include "lineal/tensor.hpp"

namespace lineal {
namespace msg {
THES_DEFINE_ENUM(SNAKE_CASE(StopCoarseningReason), thes::u8, LOWERCASE(TOO_MANY_LEVELS),
                 LOWERCASE(TOO_SMALL), LOWERCASE(TOO_SMALL_RATE))
THES_CREATE_TYPE(SNAKE_CASE(StopCoarsening), CONSTEXPR_CONSTRUCTOR,
                 MEMBERS((KEEP(reason), StopCoarseningReason)))
THES_CREATE_TYPE(SNAKE_CASE(MakeAggressive), CONSTEXPR_CONSTRUCTOR)

template<typename TSize>
struct Dimensions {
  THES_DEFINE_TYPE(SNAKE_CASE(Dimensions), CONSTEXPR_CONSTRUCTOR,
                   MEMBERS((KEEP(rows), TSize), (KEEP(columns), TSize)))
};
template<typename TGlobalSize, typename TLocalSize>
struct DistributedDimensions {
  THES_DEFINE_TYPE(SNAKE_CASE(DistributedDimensions), CONSTEXPR_CONSTRUCTOR,
                   MEMBERS((KEEP(global_size), TGlobalSize), (KEEP(local_size), TLocalSize),
                           (KEEP(own_size), TLocalSize)))
};
} // namespace msg

template<typename TLocalSize>
struct HierarchyWrightParams {
  // TODO Generalize?
  using Factor = double;

  THES_DEFINE_TYPE(SNAKE_CASE(HierarchyWrightParams), NO_CONSTRUCTOR,
                   MEMBERS((KEEP(max_level_num), std::size_t), (KEEP(min_coarsen_size), TLocalSize),
                           (KEEP(min_size_per_process), TLocalSize),
                           (KEEP(min_coarsen_factor), Factor)))
};

namespace detail {
template<typename TCoarseLhs, typename TCoarseAuxVec, typename TByteAlloc>
struct HierarchyWrightTrans;

template<SharedMatrix TCoarseLhs, SharedVector TCoarseAuxVec, typename TByteAlloc>
struct HierarchyWrightTrans<TCoarseLhs, TCoarseAuxVec, TByteAlloc> {
  using SizeByte = TCoarseLhs::SizeByte;
  using TransInfos = SimpleTransportInfo<SizeByte, TByteAlloc>;

  static TransInfos& make_trans_info(auto& trans_infos_hierarchy, auto agg_map,
                                     const auto& /*params*/, const auto& /*lhs*/,
                                     const auto& /*env*/) {
    return trans_infos_hierarchy.emplace_back(std::move(agg_map));
  }
};
} // namespace detail

template<typename TReal, AnyMatrix TCoarseLhs, AnyVector TCoarseRhs, AnyVector TCoarseSol,
         AnyVector TFinestAuxVec, AnyVector TCoarseAuxVec, StationaryIterativeSolver TPreSmoother,
         StationaryIterativeSolver TPostSmoother, typename TCoarseSolver, typename TMatAggregator,
         typename TByteAlloc>
struct HierarchyWrightInstance {
  using Real = TReal;
  using Params = HierarchyWrightParams<typename std::decay_t<TCoarseLhs>::Size>;
  using Factor = Params::Factor;
  using Trans = detail::HierarchyWrightTrans<TCoarseLhs, TCoarseAuxVec, TByteAlloc>;
  using TransInfos = Trans::TransInfos;

  static constexpr bool is_shared =
    SharedTensors<TCoarseLhs, TCoarseRhs, TCoarseSol, TFinestAuxVec, TCoarseAuxVec>;

  explicit HierarchyWrightInstance(Params params, TMatAggregator&& mat_aggregator,
                                   TPreSmoother&& pre_smoother, TPostSmoother&& post_smoother,
                                   TCoarseSolver&& coarse_solver)
      : params_(params), mat_aggregator_(std::forward<TMatAggregator>(mat_aggregator)),
        pre_smoother_(std::forward<TPreSmoother>(pre_smoother)),
        post_smoother_(std::forward<TPostSmoother>(post_smoother)),
        coarse_solver_(std::forward<TCoarseSolver>(coarse_solver)) {}

  template<AnyMatrix TFinestLhs>
  auto create(TFinestLhs&& finest_lhs, const auto& env) const {
    auto outer_env = env.add_object("create_hierarchy");
    return outer_env.add_array("levels", [&](const auto& envi) {
      return create_impl(std::forward<TFinestLhs>(finest_lhs), envi);
    });
  }

  const TMatAggregator& matrix_aggregator() const {
    return mat_aggregator_;
  }
  const TPreSmoother& pre_smoother() const {
    return pre_smoother_;
  }
  const TPostSmoother& post_smoother() const {
    return post_smoother_;
  }
  const TCoarseSolver& coarse_solver() const {
    return coarse_solver_;
  }

private:
  enum struct CoarsenAction : thes::u8 { CONTINUE, MAKE_AGGRESSIVE, STOP_COARSENING };

  CoarsenAction level_action(const AnyMatrix auto& finest_lhs, const TCoarseLhs& coarse_lhs,
                             std::size_t level_idx, bool is_aggressive, const auto& env) const {
    const std::unsigned_integral auto finest_size = tensor_size(finest_lhs);
    const std::unsigned_integral auto coarse_size = tensor_size(coarse_lhs);

    if (coarse_size < params_.min_coarsen_size) {
      env.log_named(msg::StopCoarsening{msg::StopCoarseningReason::TOO_SMALL});
      return CoarsenAction::STOP_COARSENING;
    }
    if (level_idx + 1 >= params_.max_level_num) {
      env.log_named(msg::StopCoarsening{msg::StopCoarseningReason::TOO_MANY_LEVELS});
      return CoarsenAction::STOP_COARSENING;
    }
    if (params_.min_coarsen_factor * static_cast<Factor>(coarse_size) >
        static_cast<Factor>(finest_size)) {
      if (is_aggressive) {
        env.log_named(msg::StopCoarsening{msg::StopCoarseningReason::TOO_SMALL_RATE});
        return CoarsenAction::STOP_COARSENING;
      }
      env.log_named(msg::MakeAggressive());
      return CoarsenAction::MAKE_AGGRESSIVE;
    }
    return CoarsenAction::CONTINUE;
  }

  template<typename TFinestLhs, typename TEnv>
  auto create_impl(TFinestLhs&& finest_lhs, const TEnv& env) const {
    using FinePreSmootherInst = decltype(std::declval<const TPreSmoother&>().instantiate(
      std::declval<const TFinestLhs&>(), std::declval<const TEnv&>()));
    using FinePostSmootherInst = decltype(std::declval<const TPostSmoother&>().instantiate(
      std::declval<const TFinestLhs&>(), std::declval<const TEnv&>()));
    using CoarsePreSmootherInst = decltype(std::declval<const TPreSmoother&>().instantiate(
      std::declval<const TCoarseLhs&>(), std::declval<const TEnv&>()));
    using CoarsePostSmootherInst = decltype(std::declval<const TPostSmoother&>().instantiate(
      std::declval<const TCoarseLhs&>(), std::declval<const TEnv&>()));
    using CoarsestSolverInst = decltype(std::declval<const TCoarseSolver&>().instantiate(
      std::declval<const TCoarseLhs&>(), std::declval<const TEnv&>()));

    using Hierarchy =
      Hierarchy<const TFinestLhs&, TCoarseLhs, TCoarseRhs, TCoarseSol, TFinestAuxVec, TCoarseAuxVec,
                TransInfos, FinePreSmootherInst, FinePostSmootherInst, CoarsePreSmootherInst,
                CoarsePostSmootherInst, CoarsestSolverInst>;

    using LhsHierarchy = Hierarchy::LhsHierarchy;
    using RhsHierarchy = Hierarchy::RhsHierarchy;
    using SolutionHierarchy = Hierarchy::SolutionHierarchy;

    using TransportInfoHierarchy = Hierarchy::TransportInfosHierarchy;
    using PreSmootherInstHierarchy = Hierarchy::PreSmootherInstHierarchy;
    using PostSmootherInstHierarchy = Hierarchy::PostSmootherInstHierarchy;
    using CoarseAuxHierarchy = Hierarchy::CoarseAuxHierarchy;

    static constexpr std::size_t finest_aux_num = Hierarchy::finest_aux_num;
    static constexpr std::size_t coarse_aux_num = Hierarchy::coarse_aux_num;
    static constexpr std::size_t coarsest_aux_num = Hierarchy::coarsest_aux_num;

    const std::size_t max_level_num = params_.max_level_num;

    auto lhs_hierarchy = LhsHierarchy::create_with_capacity(max_level_num);
    auto rhs_hierarchy = RhsHierarchy::create_with_capacity(max_level_num);
    auto sol_hierarchy = SolutionHierarchy::create_with_capacity(max_level_num);
    auto trans_infos_hierarchy = TransportInfoHierarchy::create_with_capacity(max_level_num);

    auto pre_smoother_hierarchy = PreSmootherInstHierarchy::create_with_capacity(max_level_num);
    auto post_smoother_hierarchy = PostSmootherInstHierarchy::create_with_capacity(max_level_num);
    auto aux_hierarchy = CoarseAuxHierarchy::create_with_capacity(max_level_num);

    std::size_t level = 0;
    bool is_aggressive = false;

    auto stop_coarsening = [&](const AnyMatrix auto& fine_lhs, const TCoarseLhs& coarse_lhs,
                               const auto& envi) {
      switch (level_action(fine_lhs, coarse_lhs, level, is_aggressive, envi)) {
        case CoarsenAction::CONTINUE: {
          return false;
        }
        case CoarsenAction::MAKE_AGGRESSIVE: {
          is_aggressive = true;
          return false;
        }
        case CoarsenAction::STOP_COARSENING: {
          return true;
        }
        default: {
          std::terminate();
        }
      }
    };

    auto coarsen = [&](const AnyMatrix auto& lhs, const auto& envi, thes::AnyBoolTag auto finest) {
      auto print_dims = [&envi](auto name, const AnyMatrix auto& mat) {
        envi.log(name, msg::Dimensions{mat.row_num(), mat.column_num()});
      };

      print_dims("fine_dims", lhs);

      TransInfos& trans_info = Trans::make_trans_info(
        trans_infos_hierarchy, mat_aggregator_.aggregate_matrix(lhs, level, is_aggressive, envi),
        params_, lhs, envi);

      const TCoarseLhs& coarse_lhs = [&]() -> const auto& {
        decltype(auto) envii = envi.add_object("coarsen_lhs");
        return lhs_hierarchy.push_back(coarsen_matrix<Real, TCoarseLhs>(lhs, trans_info, envi));
      }();

      print_dims("coarse_dims", coarse_lhs);

      rhs_hierarchy.push_back(create_numa_undef_like<TCoarseRhs>(coarse_lhs, envi));
      sol_hierarchy.push_back(create_numa_undef_like<TCoarseSol>(coarse_lhs, envi));

      const auto& finer_lhs = [&]() -> const auto& {
        if constexpr (finest) {
          return lhs;
        } else {
          return *(lhs_hierarchy.end() - 2);
        }
      }();
      const bool stop = stop_coarsening(finer_lhs, coarse_lhs, envi);

      using Tuple = std::tuple<const TCoarseLhs&, bool>;
      return Tuple{coarse_lhs, stop};
    };
    auto handle_non_coarsest = [&](const TCoarseLhs& lhs, const auto& envi) {
      {
        decltype(auto) envii = envi.add_object("instantiate_pre_smoother");
        pre_smoother_hierarchy.push_back(pre_smoother_.instantiate(lhs, envii));
      }
      {
        decltype(auto) envii = envi.add_object("instantiate_post_smoother");
        post_smoother_hierarchy.push_back(post_smoother_.instantiate(lhs, envii));
      }
      aux_hierarchy.push_back(thes::star::generate<coarse_aux_num>(
                                [&] { return create_numa_undef_like<TCoarseAuxVec>(lhs, envi); }) |
                              thes::star::to_array);
    };
    auto handle_coarsest = [&](const TCoarseLhs& lhs, const auto& envi) {
      auto pre_smoother_inst = [&] {
        decltype(auto) envii = envi.add_object("instantiate_pre_smoother");
        return pre_smoother_.instantiate(finest_lhs, envii);
      }();
      auto post_smoother_inst = [&] {
        decltype(auto) envii = envi.add_object("instantiate_post_smoother");
        return post_smoother_.instantiate(finest_lhs, envii);
      }();
      auto finest_aux = thes::star::generate<finest_aux_num>(
                          [&] { return create_numa_undef_like<TFinestAuxVec>(finest_lhs, envi); }) |
                        thes::star::to_array;
      return Hierarchy{
        std::forward<TFinestLhs>(finest_lhs),
        std::move(pre_smoother_inst),
        std::move(post_smoother_inst),
        std::move(finest_aux),
        std::move(lhs_hierarchy),
        std::move(rhs_hierarchy),
        std::move(sol_hierarchy),
        std::move(trans_infos_hierarchy),
        std::move(pre_smoother_hierarchy),
        std::move(post_smoother_hierarchy),
        std::move(aux_hierarchy),
        [&] {
          decltype(auto) envii = envi.add_object("instantiate_coarse_solver");
          return coarse_solver_.instantiate(lhs, envii);
        }(),
        thes::star::generate<coarsest_aux_num>(
          [&] { return create_undef_like<TCoarseAuxVec>(lhs); }) |
          thes::star::to_array,
      };
    };

    {
      auto envi = env.add_object();
      auto [coarse_lhs, stop] = coarsen(finest_lhs, envi, /*finest=*/thes::true_tag);
      if (stop) {
        return handle_coarsest(coarse_lhs, envi);
      }
      handle_non_coarsest(coarse_lhs, envi);
    }

    for (++level;; ++level) {
      auto envi = env.add_object();
      auto [coarse_lhs, stop] = coarsen(lhs_hierarchy.back(), envi, /*finest=*/thes::false_tag);
      if (stop) {
        return handle_coarsest(coarse_lhs, envi);
      }
      handle_non_coarsest(coarse_lhs, envi);
    }

    // TODO Should not be reachable!
    return handle_coarsest(lhs_hierarchy.back(), env);
  }

  Params params_;
  TMatAggregator mat_aggregator_;
  TPreSmoother pre_smoother_;
  TPostSmoother post_smoother_;
  TCoarseSolver coarse_solver_;
};

template<typename TReal, AnyMatrix TCoarseLhs, AnyVector TCoarseRhs, AnyVector TCoarseSol,
         AnyVector TFinestAuxVec, AnyVector TCoarseAuxVec, typename TByteAlloc>
struct HierarchyWright {
  using CoarseLhs = TCoarseLhs;
  using Params = HierarchyWrightParams<typename CoarseLhs::Size>;

  explicit HierarchyWright(Params params) : params_(params) {}

  template<StationaryIterativeSolver TPreSmoother, StationaryIterativeSolver TPostSmoother,
           DirectSolver TCoarseSolver, typename TMatAggregator>
  auto instantiate(TMatAggregator&& mat_aggregator, TPreSmoother&& pre_smoother,
                   TPostSmoother&& post_smoother, TCoarseSolver&& coarse_solver) const {
    using Instance =
      HierarchyWrightInstance<TReal, TCoarseLhs, TCoarseRhs, TCoarseSol, TFinestAuxVec,
                              TCoarseAuxVec, TPreSmoother, TPostSmoother, TCoarseSolver,
                              TMatAggregator, TByteAlloc>;
    return Instance{
      params_,
      std::forward<TMatAggregator>(mat_aggregator),
      std::forward<TPreSmoother>(pre_smoother),
      std::forward<TPostSmoother>(post_smoother),
      std::forward<TCoarseSolver>(coarse_solver),
    };
  }

private:
  Params params_;
};
} // namespace lineal

#endif // INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_HIERARCHY_WRIGHT_HPP
