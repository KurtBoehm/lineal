// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_LINEAR_SOLVER_BICGSTAB_SOLVER_HPP
#define INCLUDE_LINEAL_LINEAR_SOLVER_BICGSTAB_SOLVER_HPP

#include <array>
#include <cassert>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

#include "thesauros/static-ranges.hpp"
#include "thesauros/types.hpp"
#include "thesauros/utility.hpp"

#include "lineal/base.hpp"
#include "lineal/component-wise.hpp"
#include "lineal/tensor.hpp"

namespace lineal {
template<typename TDefs, typename TRecomputeResidualController,
         OptStationaryIterativeSolver TPrecon = void>
struct BiCgStabSolver : public SharedNonStationaryIterativeSolverBase,
                        public DistributedNonStationaryIterativeSolverBase {
  using Defs = TDefs;
  using Real = Defs::Real;
  using LoReal = Defs::LoReal;
  using HiVector = Defs::HiVector;
  using LoVector = Defs::LoVector;
  using Precon = std::decay_t<TPrecon>;
  using PreconStorage = thes::VoidStorage<TPrecon>;
  static constexpr bool has_preconditioner = !std::is_void_v<TPrecon>;

  template<AnyMatrix TLhs, typename TPreconInst>
  struct Instance {
    using Lhs = std::decay_t<TLhs>;
    using PreconInst = std::decay_t<TPreconInst>;
    using PreconInstStorage = thes::VoidStorage<TPreconInst>;
    static constexpr bool has_precon_inst = !std::is_void_v<TPreconInst>;

    template<AnyVector TSol, AnyVector TRhs>
    struct SystemInstance {
      using Sol = std::decay_t<TSol>;
      using SolValue = Sol::Value;

      SystemInstance(Instance& parent, TSol&& sol, TRhs&& rhs, const Env auto& env)
          : parent_(&parent), sol_(std::forward<TSol>(sol)), rhs_(std::forward<TRhs>(rhs)),
            r_(create_numa_undef_like<HiVector>(parent.lhs_, env)),
            r0_(create_numa_undef_like<HiVector>(parent.lhs_, env)),
            p_(create_numa_undef_like<LoVector>(parent.lhs_, env)),
            v_(create_numa_undef_like<LoVector>(parent.lhs_, env)),
            t_(create_numa_undef_like<LoVector>(parent.lhs_, env)), px_([&] {
              if constexpr (has_preconditioner) {
                return create_numa_undef_like<LoVector>(parent.lhs_, env);
              } else {
                return thes::Empty{};
              }
            }()),
            precon_aux_(compute_aux(parent.lhs_, env)) {
        const auto& expo = env.execution_policy();

        const auto res = subtract<Real>(rhs_, multiply<Real>(parent_->lhs_, sol_));
        res_norm_sq_ = rho_ =
          euclidean_squared<Real>(assign_expr(p_, assign_expr(r0_, assign_expr(r_, res))), expo);
        assert(std::isfinite(res_norm_sq_));
      }

      void iterate(const Env auto& env) {
        assert(iteration_ < std::numeric_limits<std::size_t>::max());

        const auto& expo = env.execution_policy();
        const auto& dist_info = distributed_info_storage(sol_);

        {
          auto& y = [&]() -> auto& {
            if constexpr (has_preconditioner) {
              parent_->precon_inst_.apply(constant_like(sol_, compat::zero<SolValue>()), px_, p_,
                                          precon_aux_, env);
              return px_;
            } else {
              return p_;
            }
          }();

          const auto alpha_denom =
            dot<Real>(r0_, assign_expr(v_, multiply<Real>(parent_->lhs_, y)), expo);
          assert(std::isfinite(alpha_denom) && alpha_denom != 0);
          alpha_ = rho_ / alpha_denom;
          assert(std::isfinite(alpha_));

          assign(sol_, add<Real>(sol_, scale<Real>(y, alpha_)), expo);

          res_norm_sq_ = euclidean_squared<Real>(
            assign_expr(r_, subtract<Real>(r_, scale<Real>(v_, alpha_))), expo);
          assert(std::isfinite(res_norm_sq_));
        }

        {
          auto& z = [&]() -> auto& {
            if constexpr (has_preconditioner) {
              parent_->precon_inst_.apply(constant_like(sol_, compat::zero<SolValue>()), px_, r_,
                                          precon_aux_, env);
              return px_;
            } else {
              return r_;
            }
          }();

          const auto [omega_num, omega_den] = multiconsume(
            thes::Tuple{
              dot_consumer<Real, 0, 1>(dist_info),
              euclidean_squared_consumer<Real, 0>(dist_info),
            },
            thes::Tuple{assign_expr(t_, multiply<Real>(parent_->lhs_, z)), r_}, expo);
          assert(std::isfinite(omega_num) && std::isfinite(omega_den) && omega_den != 0);
          omega_ = omega_num / omega_den;
          assert(std::isfinite(omega_));

          assign(sol_, add<Real>(sol_, scale<Real>(z, omega_)), expo);
        }

        {
          const auto old_rho = rho_;
          auto res_op = [&](auto res) {
            const auto [rho, res_norm_sq] = multiconsume(
              thes::Tuple{
                dot_consumer<Real, 0, 1>(dist_info),
                euclidean_squared_consumer<Real, 0>(dist_info),
              },
              thes::Tuple{assign_expr(r_, res), r0_}, expo);
            rho_ = rho;
            res_norm_sq_ = res_norm_sq;
          };
          if (parent_->rere_controller_(iteration_, thes::fast::sqrt(res_norm_sq_), env)) {
            res_op(subtract<Real>(rhs_, multiply<Real>(parent_->lhs_, sol_)));
          } else {
            res_op(subtract<Real>(r_, scale<Real>(t_, omega_)));
          }
          assert(std::isfinite(res_norm_sq_));

          const Real beta = (rho_ / old_rho) * (alpha_ / omega_);
          assign(p_, add<Real>(r_, scale(subtract<Real>(p_, scale(v_, omega_)), beta)), expo);
        }

        ++iteration_;
      }

      [[nodiscard]] Real residual_norm() const {
        return thes::fast::sqrt(res_norm_sq_);
      }
      [[nodiscard]] Real rho() const {
        return rho_;
      }
      [[nodiscard]] const TSol& solution() const {
        return sol_;
      }

    private:
      static constexpr std::size_t aux_size = [] {
        if constexpr (has_precon_inst) {
          return PreconInst::ex_situ_aux_size;
        } else {
          return 0;
        }
      }();

      static auto compute_aux(AnyMatrix auto& lhs, const Env auto& env) {
        return thes::star::generate<LoVector, aux_size>(
                 [&]() { return create_numa_undef_like<LoVector>(lhs, env); }) |
               thes::star::to_array;
      }

      Instance* parent_;
      TSol sol_;
      TRhs rhs_;

      std::size_t iteration_{0};
      HiVector r_;
      HiVector r0_;
      LoVector p_;
      LoVector v_;
      LoVector t_;
      std::conditional_t<has_preconditioner, LoVector, thes::Empty> px_;
      std::array<LoVector, aux_size> precon_aux_;

      Real res_norm_sq_{std::numeric_limits<Real>::signaling_NaN()};
      Real rho_{std::numeric_limits<Real>::signaling_NaN()};
      Real alpha_{std::numeric_limits<Real>::signaling_NaN()};
      Real omega_{std::numeric_limits<Real>::signaling_NaN()};
    };

    Instance(TLhs&& lhs, const TRecomputeResidualController& rere_controller,
             const PreconStorage& precon, const Env auto& env)
        : lhs_(std::forward<TLhs>(lhs)), rere_controller_(rere_controller),
          precon_inst_(precon.instantiate(lhs, env)) {}

    Instance(TLhs&& lhs, const TRecomputeResidualController& rere_controller,
             PreconInstStorage&& precon_inst)
        : lhs_(std::forward<TLhs>(lhs)), rere_controller_(rere_controller),
          precon_inst_(std::forward<PreconInstStorage>(precon_inst)) {}

    Instance(TLhs&& lhs, const TRecomputeResidualController& rere_controller)
    requires(!has_precon_inst)
        : lhs_(std::forward<TLhs>(lhs)), rere_controller_(rere_controller) {}

    template<AnyVector TSol, AnyVector TRhs>
    SystemInstance<TSol, TRhs> instantiate(TSol&& sol, TRhs&& rhs, const Env auto& env) {
      return {*this, std::forward<TSol>(sol), std::forward<TRhs>(rhs), env};
    }

  private:
    TLhs lhs_;
    const TRecomputeResidualController& rere_controller_;
    [[no_unique_address]] PreconInstStorage precon_inst_{};
  };

  template<AnyMatrix TLhs, typename TPreconInst>
  auto instantiate(TLhs lhs, TPreconInst&& precon_instance, const Env auto& /*env*/) {
    return Instance<TLhs, TPreconInst>{
      std::forward<TLhs>(lhs),
      rere_controller_,
      std::forward<TPreconInst>(precon_instance),
    };
  }

  template<AnyMatrix TLhs>
  auto instantiate(TLhs&& lhs, const Env auto& env) const {
    if constexpr (has_preconditioner) {
      using PreconInst = decltype(precon_.instantiate(lhs, env));
      return Instance<TLhs, PreconInst>{std::forward<TLhs>(lhs), rere_controller_, precon_, env};
    } else {
      return Instance<TLhs, void>{std::forward<TLhs>(lhs), rere_controller_};
    }
  }

  explicit BiCgStabSolver(thes::TypeTag<TDefs> /*tag*/,
                          TRecomputeResidualController rere_controller, PreconStorage&& precon)
      : rere_controller_(std::move(rere_controller)), precon_(std::forward<PreconStorage>(precon)) {
  }
  explicit BiCgStabSolver(TRecomputeResidualController rere_controller, PreconStorage&& precon)
      : rere_controller_(std::move(rere_controller)), precon_(std::forward<PreconStorage>(precon)) {
  }

  BiCgStabSolver(thes::TypeTag<TDefs> /*tag*/, TRecomputeResidualController rere_controller)
  requires(!has_preconditioner)
      : rere_controller_(std::move(rere_controller)) {}
  explicit BiCgStabSolver(TRecomputeResidualController rere_controller)
  requires(!has_preconditioner)
      : rere_controller_(std::move(rere_controller)) {}

private:
  TRecomputeResidualController rere_controller_;
  [[no_unique_address]] PreconStorage precon_{};
};
template<typename TDefs, typename TRecomputeResidualController,
         OptStationaryIterativeSolver TPrecon>
BiCgStabSolver(thes::TypeTag<TDefs>, TRecomputeResidualController, TPrecon&&)
  -> BiCgStabSolver<TDefs, TRecomputeResidualController, TPrecon>;
} // namespace lineal

#endif // INCLUDE_LINEAL_LINEAR_SOLVER_BICGSTAB_SOLVER_HPP
