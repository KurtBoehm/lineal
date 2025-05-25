// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_SOLVER_HPP
#define INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_SOLVER_HPP

#include <cassert>
#include <cstddef>
#include <type_traits>

#include "thesauros/containers.hpp"

#include "lineal/base.hpp"
#include "lineal/component-wise/vector-expression.hpp"
#include "lineal/linear-solver/multigrid/transport.hpp"

namespace lineal {
template<typename TCycle, typename THierarchyWrightInstance>
struct MultiGridSolver : public SharedStationaryIterativeSolverBase,
                         public DistributedStationaryIterativeSolverBase {
  using HierarchyWrightInstance = std::decay_t<THierarchyWrightInstance>;
  using Real = HierarchyWrightInstance::Real;
  using CycleInfo = TCycle::CycleInfo;
  static constexpr bool is_shared = HierarchyWrightInstance::is_shared;

  template<typename THierarchy>
  struct Instance {
    static constexpr std::size_t in_situ_aux_size = 0;
    static constexpr std::size_t ex_situ_aux_size = 0;

    Instance(Real inv_relax, TCycle cycle, THierarchy&& hierarchy)
        : inv_relax_(inv_relax), cycle_(cycle), hierarchy_(std::forward<THierarchy>(hierarchy)) {}

    void apply(AnyVector auto& sol, const AnyVector auto& rhs, const auto& /*aux*/, auto& env) {
      apply_impl(
        sol, rhs, [&](auto& pre, auto& aux, const auto& envi) { pre.apply(sol, rhs, aux, envi); },
        env);
    }

    void apply(const AnyVector auto& in_sol, AnyVector auto& out_sol, const AnyVector auto& rhs,
               const auto& /*aux*/, auto& env) {
      apply_impl(
        out_sol, rhs,
        [&](auto& pre, auto& aux, const auto& envi) { pre.apply(in_sol, out_sol, rhs, aux, envi); },
        env);
    }

  private:
    void apply_impl(AnyVector auto& sol, const AnyVector auto& rhs, auto pre_smooth_finest,
                    auto& env) {
      const auto& expo = env.execution_policy();

      const std::size_t level_num = hierarchy_.level_num();
      auto cycle_infos = thes::FixedAllocArray<CycleInfo>::create_with_capacity(level_num);
      cycle_infos.push_back(cycle_.fine_cycle_info());

      while (!cycle_infos.empty()) {
        const std::size_t level = cycle_infos.size() - 1;
        const bool is_finest = level == 0;
        const bool is_second_coarsest = level + 2 == level_num;
        const bool is_coarsest = level + 1 == level_num;
        CycleInfo& info = cycle_infos.back();

        if (is_coarsest) {
          // Directly solve on the coarsest level
          assert(info.is_end());

          auto& solver = hierarchy_.coarsest_solver_instance();
          solver.solve(hierarchy_.coarsest_sol(), hierarchy_.coarsest_rhs(),
                       hierarchy_.coarsest_aux(), env);

          cycle_infos.pop_back();
          continue;
        }

        if (info.is_begin()) {
          // Downward step
          auto& coarse_trans_infos = hierarchy_.coarse_trans_info(level);
          auto& coarse_rhs = hierarchy_.coarse_rhs(level);

          if (is_finest) {
            const auto& fine_lhs = hierarchy_.finest_lhs();
            auto& pre_smoother = hierarchy_.finest_pre_smoother();
            auto& aux = hierarchy_.finest_aux();

            pre_smooth_finest(pre_smoother, aux, env);
            thes::fancy_visit(
              [&](auto& trans) {
                coarsen_vector<Real>(coarse_rhs, subtract<Real>(rhs, multiply<Real>(fine_lhs, sol)),
                                     trans, env);
              },
              coarse_trans_infos);
          } else {
            const auto& fine_lhs = hierarchy_.fine_lhs(level);
            auto& fine_sol = hierarchy_.fine_solution(level);
            auto& fine_rhs = hierarchy_.fine_rhs(level);
            auto& fine_pre_smoother = hierarchy_.fine_pre_smoother(level);
            auto& fine_aux = hierarchy_.fine_aux(level);

            // The first argument replaces setting the coarse solution to 0 before recursion
            fine_pre_smoother.apply(constant_like(fine_sol, Real{0}), fine_sol, fine_rhs, fine_aux,
                                    env);
            thes::fancy_visit(
              [&](auto& trans) {
                coarsen_vector<Real>(coarse_rhs,
                                     subtract<Real>(fine_rhs, multiply<Real>(fine_lhs, fine_sol)),
                                     trans, env);
              },
              coarse_trans_infos);
          }
        }

        if (info.is_end()) {
          // Upward step
          auto& coarse_trans_infos = hierarchy_.coarse_trans_info(level);
          auto& coarse_sol = hierarchy_.coarse_solution(level);

          auto worker = [&](auto& fine_sol, const auto& fine_rhs, const auto& fine_lhs,
                            auto& post_smoother, auto& aux) {
            thes::fancy_visit(
              [&](auto& trans) {
                auto refined = [&] {
                  if constexpr (is_shared) {
                    return refine_vector(coarse_sol, trans, env);
                  } else {
                    return refine_vector(coarse_sol, fine_lhs.distributed_info(), trans, env);
                  }
                }();
                assign(fine_sol, add<Real>(fine_sol, scale<Real>(refined, inv_relax_)), expo);
              },
              coarse_trans_infos);

            post_smoother.apply(fine_sol, fine_rhs, aux, env);
          };

          if (is_finest) {
            const auto& fine_lhs = hierarchy_.finest_lhs();
            auto& post_smoother = hierarchy_.finest_post_smoother();
            auto& aux = hierarchy_.finest_aux();

            worker(sol, rhs, fine_lhs, post_smoother, aux);
          } else {
            const auto& fine_rhs = hierarchy_.fine_rhs(level);
            const auto& fine_lhs = hierarchy_.fine_lhs(level);
            auto& fine_sol = hierarchy_.fine_solution(level);
            auto& post_smoother = hierarchy_.fine_post_smoother(level);
            auto& aux = hierarchy_.fine_aux(level);

            worker(fine_sol, fine_rhs, fine_lhs, post_smoother, aux);
          }
        }

        if (info.is_end()) {
          // Remove itself from the stack
          cycle_infos.pop_back();
        } else {
          ++info;

          // Add coarser level to stack
          if (is_second_coarsest) {
            // Add coarsest level
            cycle_infos.push_back(cycle_.get_coarsest());
          } else {
            // Add in-between level
            cycle_infos.push_back(info.get_coarser());
          }
        }
      }
    }

    Real inv_relax_;
    TCycle cycle_;
    THierarchy hierarchy_;
  };

  MultiGridSolver(Real relax, TCycle cycle, THierarchyWrightInstance&& hierarchy_wright_instance)
      : inv_relax_(Real{1} / relax), cycle_(cycle),
        hierarchy_wright_instance_(
          std::forward<THierarchyWrightInstance>(hierarchy_wright_instance)) {}

  template<AnyMatrix TLhs>
  auto instantiate(TLhs&& lhs, const auto& env) const {
    auto create_hierarchy = [&] {
      return hierarchy_wright_instance_.create(std::forward<TLhs>(lhs), env);
    };

    using Hierarchy = decltype(create_hierarchy());
    return Instance<Hierarchy>(inv_relax_, cycle_, create_hierarchy());
  }

private:
  Real inv_relax_;
  TCycle cycle_;
  THierarchyWrightInstance hierarchy_wright_instance_;
};
template<typename TCycle, typename THierarchyWrightInstance>
MultiGridSolver(typename std::decay_t<THierarchyWrightInstance>::Real, TCycle&&,
                THierarchyWrightInstance&&) -> MultiGridSolver<TCycle, THierarchyWrightInstance>;
} // namespace lineal

#endif // INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_SOLVER_HPP
