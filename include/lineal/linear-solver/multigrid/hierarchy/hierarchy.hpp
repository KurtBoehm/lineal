// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_HIERARCHY_HIERARCHY_HPP
#define INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_HIERARCHY_HIERARCHY_HPP

#include <array>
#include <cassert>
#include <cstddef>
#include <type_traits>

#include "thesauros/containers.hpp"

#include "lineal/base.hpp"

namespace lineal {
// TODO Assumes that the solver instances are always applied in situ, with the exception
// of the coarse-level pre-smoother, which is applied ex situ.
// Levels are numbered starting at 0 for the finest level.
template<AnyMatrix TFinestLhs, AnyMatrix TCoarseLhs, AnyVector TCoarseRhs, AnyVector TCoarseSol,
         AnyVector TFinestAuxVec, AnyVector TCoarseAuxVec, TransportInfos TTransportInfos,
         typename TFinestPreSmootherInst, typename TFinestPostSmootherInst,
         typename TCoarsePreSmootherInst, typename TCoarsePostSmootherInst,
         typename TCoarsestSolverInst>
struct Hierarchy {
  using FinestLhs = TFinestLhs;
  using FinestPreSmootherInstance = TFinestPreSmootherInst;
  using FinestPostSmootherInstance = TFinestPostSmootherInst;

  using LhsHierarchy = thes::FixedAllocArray<TCoarseLhs>;
  using RhsHierarchy = thes::FixedAllocArray<TCoarseRhs>;
  using SolutionHierarchy = thes::FixedAllocArray<TCoarseSol>;
  using TransportInfosHierarchy = thes::FixedAllocArray<TTransportInfos>;

  using PreSmootherInstHierarchy = thes::FixedAllocArray<TCoarsePreSmootherInst>;
  using PostSmootherInstHierarchy = thes::FixedAllocArray<TCoarsePostSmootherInst>;

  static constexpr std::size_t finest_aux_num =
    std::max(std::decay_t<TFinestPreSmootherInst>::in_situ_aux_size,
             std::decay_t<TFinestPostSmootherInst>::in_situ_aux_size);
  static constexpr std::size_t coarse_aux_num =
    std::max(std::decay_t<TCoarsePreSmootherInst>::ex_situ_aux_size,
             std::decay_t<TCoarsePostSmootherInst>::in_situ_aux_size);
  static constexpr std::size_t coarsest_aux_num = std::decay_t<TCoarsestSolverInst>::aux_size;

  using FinestAuxVector = TFinestAuxVec;
  using CoarseAuxVector = TCoarseAuxVec;
  using FinestAux = std::array<FinestAuxVector, finest_aux_num>;
  using CoarseAux = std::array<CoarseAuxVector, coarse_aux_num>;
  using CoarseAuxHierarchy = thes::FixedAllocArray<CoarseAux>;
  using CoarsestAux = std::array<CoarseAuxVector, coarsest_aux_num>;

  Hierarchy(FinestLhs&& finest_lhs, FinestPreSmootherInstance&& finest_pre_smoother,
            FinestPostSmootherInstance&& finest_post_smoother, FinestAux&& finest_aux,
            LhsHierarchy&& lhs_hierarchy, RhsHierarchy&& rhs_hierarchy,
            SolutionHierarchy&& sol_hierarchy, TransportInfosHierarchy&& trans_infos_hierarchy,
            PreSmootherInstHierarchy&& pre_smoother_hierarchy,
            PostSmootherInstHierarchy&& post_smoother_hierarchy, CoarseAuxHierarchy&& aux_hierarchy,
            TCoarsestSolverInst&& coarsest_solver, CoarsestAux&& coarsest_aux)
      : finest_lhs_(std::forward<FinestLhs>(finest_lhs)),
        finest_pre_smoother_(std::forward<FinestPreSmootherInstance>(finest_pre_smoother)),
        finest_post_smoother_(std::forward<FinestPostSmootherInstance>(finest_post_smoother)),
        finest_aux_(std::forward<FinestAux>(finest_aux)),
        lhs_hierarchy_(std::forward<LhsHierarchy>(lhs_hierarchy)),
        rhs_hierarchy_(std::forward<RhsHierarchy>(rhs_hierarchy)),
        sol_hierarchy_(std::forward<SolutionHierarchy>(sol_hierarchy)),
        trans_infos_hierarchy_(std::forward<TransportInfosHierarchy>(trans_infos_hierarchy)),
        pre_smoother_hierarchy_(std::forward<PreSmootherInstHierarchy>(pre_smoother_hierarchy)),
        post_smoother_hierarchy_(std::forward<PostSmootherInstHierarchy>(post_smoother_hierarchy)),
        aux_hierarchy_(std::forward<CoarseAuxHierarchy>(aux_hierarchy)),
        coarsest_solver_(std::forward<TCoarsestSolverInst>(coarsest_solver)),
        coarsest_aux_(std::forward<CoarsestAux>(coarsest_aux)) {}

  [[nodiscard]] std::size_t level_num() const {
    assert(lhs_hierarchy_.size() == rhs_hierarchy_.size());
    assert(lhs_hierarchy_.size() == sol_hierarchy_.size());
    assert(lhs_hierarchy_.size() == trans_infos_hierarchy_.size());
    assert(lhs_hierarchy_.size() == pre_smoother_hierarchy_.size() + 1);
    assert(lhs_hierarchy_.size() == post_smoother_hierarchy_.size() + 1);
    assert(lhs_hierarchy_.size() == aux_hierarchy_.size() + 1);
    return lhs_hierarchy_.size() + 1;
  }

  // finest

  const FinestLhs& finest_lhs() const {
    return finest_lhs_;
  }
  FinestPreSmootherInstance& finest_pre_smoother() {
    return finest_pre_smoother_;
  }
  FinestPostSmootherInstance& finest_post_smoother() {
    return finest_post_smoother_;
  }
  FinestAux& finest_aux() {
    return finest_aux_;
  }

  // hierarchy

  const TCoarseLhs& fine_lhs(std::size_t level) const {
    assert(level > 0);
    return lhs_hierarchy_[level - 1];
  }
  TCoarseRhs& coarse_rhs(std::size_t level) {
    return rhs_hierarchy_[level];
  }
  const TCoarseRhs& fine_rhs(std::size_t level) const {
    assert(level > 0);
    return rhs_hierarchy_[level - 1];
  }
  TCoarseSol& fine_solution(std::size_t level) {
    assert(level > 0);
    return sol_hierarchy_[level - 1];
  }
  TCoarseSol& coarse_solution(std::size_t level) {
    return sol_hierarchy_[level];
  }

  TCoarsePreSmootherInst& fine_pre_smoother(std::size_t level) {
    assert(level > 0);
    return pre_smoother_hierarchy_[level - 1];
  }
  TCoarsePostSmootherInst& fine_post_smoother(std::size_t level) {
    assert(level > 0);
    return post_smoother_hierarchy_[level - 1];
  }

  CoarseAux& fine_aux(std::size_t level) {
    assert(level > 0);
    return aux_hierarchy_[level - 1];
  }

  TTransportInfos& coarse_trans_info(std::size_t level) {
    return trans_infos_hierarchy_[level];
  }

  // coarsest

  [[nodiscard]] const TCoarseLhs& coarsest_lhs() const {
    return lhs_hierarchy_.back();
  }
  [[nodiscard]] const TCoarseRhs& coarsest_rhs() const {
    return rhs_hierarchy_.back();
  }
  [[nodiscard]] TCoarseSol& coarsest_sol() {
    return sol_hierarchy_.back();
  }

  [[nodiscard]] TCoarsestSolverInst& coarsest_solver_instance() {
    return coarsest_solver_;
  }

  [[nodiscard]] CoarsestAux& coarsest_aux() {
    return coarsest_aux_;
  }

private:
  // Only on the finest level
  FinestLhs finest_lhs_;
  FinestPreSmootherInstance finest_pre_smoother_;
  FinestPostSmootherInstance finest_post_smoother_;
  FinestAux finest_aux_;

  // On all but the finest level
  LhsHierarchy lhs_hierarchy_{};
  RhsHierarchy rhs_hierarchy_{};
  SolutionHierarchy sol_hierarchy_{};
  TransportInfosHierarchy trans_infos_hierarchy_{};

  // On all but the finest and coarsest level
  PreSmootherInstHierarchy pre_smoother_hierarchy_{};
  PostSmootherInstHierarchy post_smoother_hierarchy_{};
  CoarseAuxHierarchy aux_hierarchy_{};

  // Only on the coarsest level
  TCoarsestSolverInst coarsest_solver_;
  CoarsestAux coarsest_aux_;
};
} // namespace lineal

#endif // INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_HIERARCHY_HIERARCHY_HPP
