// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_LINEAR_SOLVER_ITERATIVE_STATIONARY_BROOM_HPP
#define INCLUDE_LINEAL_LINEAR_SOLVER_ITERATIVE_STATIONARY_BROOM_HPP

#include <cassert>
#include <cstddef>
#include <optional>
#include <type_traits>
#include <utility>

#include "thesauros/types.hpp"

#include "lineal/base.hpp"
#include "lineal/component-wise.hpp"
#include "lineal/parallel/distributed-info.hpp"

#ifndef NDEBUG
#include <limits>
#endif

namespace lineal {
template<typename TBroom>
struct SingleSweepImpl {
  using Broom = TBroom;
  static constexpr std::size_t managed_in_situ_aux_size = 1;
  static constexpr std::size_t managed_ex_situ_aux_size = 1;
  static constexpr std::size_t unmanaged_in_situ_aux_size = 1;
  static constexpr std::size_t unmanaged_ex_situ_aux_size = 0;

  template<typename TSelf, AnyVector TSolIn, AnyVector TSolOut, typename TAux>
  static void apply(TSelf& self, TSolIn&& sol_in, TSolOut& sol_out, const AnyVector auto& rhs,
                    const TAux& /*aux*/, const Env auto& env)
  requires(std::is_void_v<typename TSelf::IterManager> && thes::star::size<TAux> >= 0)
  {
    return TBroom::sweep(self, std::forward<TSolIn>(sol_in), sol_out, rhs, env);
  }

  template<typename TSelf, AnyVector TSol, typename TAux>
  static void apply(TSelf& self, TSol& sol, const AnyVector auto& rhs, TAux&& aux,
                    const Env auto& env)
  requires(std::is_void_v<typename TSelf::IterManager> && thes::star::size<TAux> >= 1)
  {
    AnyVector decltype(auto) aux_vec = thes::star::get_at<0>(std::forward<TAux>(aux));
    swap(sol, aux_vec, env.execution_policy());
    return TBroom::sweep(self, aux_vec, sol, rhs, env);
  }

  template<typename TSelf, AnyVector TSolIn, AnyVector TSolOut, typename TAux>
  static void apply(TSelf& self, TSolIn&& sol_in, TSolOut& sol_out, const AnyVector auto& rhs,
                    TAux&& aux, const Env auto& env)
  requires(!std::is_void_v<typename TSelf::IterManager> && thes::star::size<TAux> >= 1)
  {
    AnyVector decltype(auto) aux_vec = thes::star::get_at<0>(std::forward<TAux>(aux));
    const auto iter_num = self.iter_manager_.iteration_num();
    using IterSize = std::decay_t<decltype(iter_num)>;

    if (iter_num == 0) {
      assign(sol_out, sol_in, env.execution_policy());
      return;
    }

    IterSize iter = 0;
#ifndef NDEBUG
    using AuxVal = std::decay_t<decltype(aux_vec)>::Value;
    assign(aux_vec, constant_like(aux_vec, std::numeric_limits<AuxVal>::signaling_NaN()),
           env.execution_policy());
    using OutVal = std::decay_t<TSolOut>::Value;
    assign(sol_out, constant_like(sol_out, std::numeric_limits<OutVal>::signaling_NaN()),
           env.execution_policy());
#endif
    if (iter_num % 2 == 0) {
      TBroom::sweep(self, std::forward<TSolIn>(sol_in), aux_vec, rhs, env);
      TBroom::sweep(self, aux_vec, sol_out, rhs, env);
      iter += 2;
    } else {
      TBroom::sweep(self, std::forward<TSolIn>(sol_in), sol_out, rhs, env);
      ++iter;
    }
    for (; iter < iter_num; iter += 2) {
      TBroom::sweep(self, sol_out, aux_vec, rhs, env);
      TBroom::sweep(self, aux_vec, sol_out, rhs, env);
    }
  }

  template<typename TSelf, AnyVector TSol, typename TAux>
  static void apply(TSelf& self, TSol& sol, const AnyVector auto& rhs, TAux&& aux,
                    const Env auto& env)
  requires(!std::is_void_v<typename TSelf::IterManager> && thes::star::size<TAux> >= 1)
  {
    AnyVector decltype(auto) aux_vec = thes::star::get_at<0>(std::forward<TAux>(aux));
    const auto iter_num = self.iter_manager_.iteration_num();
    using IterSize = std::decay_t<decltype(iter_num)>;

    if (iter_num == 0) {
      return;
    }

    IterSize iter = 0;
    for (; iter + 2 <= iter_num; iter += 2) {
#ifndef NDEBUG
      using AuxVal = std::decay_t<decltype(aux_vec)>::Value;
      assign(aux_vec, constant_like(aux_vec, std::numeric_limits<AuxVal>::signaling_NaN()),
             env.execution_policy());
#endif
      TBroom::sweep(self, sol, aux_vec, rhs, env);
#ifndef NDEBUG
      using SolVal = std::decay_t<TSol>::Value;
      assign(sol, constant_like(sol, std::numeric_limits<SolVal>::signaling_NaN()),
             env.execution_policy());
#endif
      TBroom::sweep(self, aux_vec, sol, rhs, env);
    }
    if (iter != iter_num) {
      assert(iter + 1 == iter_num);
      TBroom::sweep(self, sol, aux_vec, rhs, env);
      swap(sol, aux_vec, env.execution_policy());
    }
  }
};

template<typename TBroom>
struct DoubleSweepImpl {
  using Broom = TBroom;
  static constexpr std::size_t managed_in_situ_aux_size = 1;
  static constexpr std::size_t managed_ex_situ_aux_size = 1;
  static constexpr std::size_t unmanaged_in_situ_aux_size = 1;
  static constexpr std::size_t unmanaged_ex_situ_aux_size = 1;

  template<typename TSelf, AnyVector TSolIn, AnyVector TSolOut, typename TAux>
  static void apply(TSelf& self, TSolIn&& sol_in, TSolOut& sol_out, const AnyVector auto& rhs,
                    TAux&& aux, const Env auto& env)
  requires(std::is_void_v<typename TSelf::IterManager> && thes::star::size<TAux> >= 0)
  {
    return TBroom::sweep(self, std::forward<TSolIn>(sol_in), sol_out, rhs, std::forward<TAux>(aux),
                         env);
  }

  template<typename TSelf, AnyVector TSol, typename TAux>
  static void apply(TSelf& self, TSol& sol, const AnyVector auto& rhs, TAux&& aux,
                    const Env auto& env)
  requires(std::is_void_v<typename TSelf::IterManager> && thes::star::size<TAux> >= 1)
  {
    return TBroom::sweep(self, sol, sol, rhs, std::forward<TAux>(aux), env);
  }

  template<typename TSelf, AnyVector TSolIn, AnyVector TSolOut, typename TAux>
  static void apply(TSelf& self, TSolIn&& sol_in, TSolOut& sol_out, const AnyVector auto& rhs,
                    TAux&& aux, const Env auto& env)
  requires(!std::is_void_v<typename TSelf::IterManager> && thes::star::size<TAux> >= 1)
  {
    const auto iter_num = self.iter_manager_.iteration_num();

    if (iter_num == 0) {
      const auto& expo = env.execution_policy();
      assign(sol_out, std::forward<TSolIn>(sol_in), expo);
      return;
    }

    TBroom::sweep(self, std::forward<TSolIn>(sol_in), sol_out, rhs, aux, env);
    for ([[maybe_unused]] const auto iter : thes::range(iter_num - 1)) {
      TBroom::sweep(self, sol_out, sol_out, rhs, aux, env);
    }
  }

  template<typename TSelf, AnyVector TSol, typename TAux>
  static void apply(TSelf& self, TSol& sol, const AnyVector auto& rhs, TAux&& aux,
                    const Env auto& env)
  requires(!std::is_void_v<typename TSelf::IterManager> && thes::star::size<TAux> >= 1)
  {
    for ([[maybe_unused]] const auto iter : thes::range(self.iter_manager_.iteration_num())) {
      TBroom::sweep(self, sol, sol, rhs, aux, env);
    }
  }
};

template<typename TReal, typename TImpl, typename TIterManager>
struct RelaxedSweepSolver : public SharedStationaryIterativeSolverBase,
                            public DistributedStationaryIterativeSolverBase {
  using Real = TReal;
  using IterManager = TIterManager;
  using IterManagerStorage = thes::VoidStorage<IterManager>;

  template<AnyMatrix TLhs>
  struct Instance {
    friend TImpl;
    friend TImpl::Broom;

    using Real = TReal;
    using Lhs = std::decay_t<TLhs>;

    using DistributedInfo = DistributedInfoOf<Lhs>;
    using IterManager = thes::VoidConstLvalRef<RelaxedSweepSolver::IterManager>;
    using IterManagerStorage = thes::VoidStorage<IterManager>;
    static constexpr bool is_managed = !std::is_void_v<IterManager>;

    static constexpr std::size_t in_situ_aux_size =
      is_managed ? TImpl::managed_in_situ_aux_size : TImpl::unmanaged_in_situ_aux_size;
    static constexpr std::size_t ex_situ_aux_size =
      is_managed ? TImpl::managed_ex_situ_aux_size : TImpl::unmanaged_ex_situ_aux_size;

    Instance(TLhs&& lhs, Real relax, std::optional<std::size_t> min_per_thread,
             IterManagerStorage iter_manager)
        : lhs_(std::forward<TLhs>(lhs)), relax_(relax), min_per_thread_(min_per_thread),
          iter_manager_(iter_manager) {}

    template<typename... TArgs>
    void apply(TArgs&&... args)
    requires(requires { TImpl::apply(*this, std::forward<TArgs>(args)...); })
    {
      TImpl::apply(*this, std::forward<TArgs>(args)...);
    }

    [[nodiscard]] const TLhs& lhs() const {
      return lhs_;
    }
    [[nodiscard]] Real relax() const {
      return relax_;
    }

  private:
    TLhs lhs_;
    Real relax_;
    std::optional<std::size_t> min_per_thread_;
    [[no_unique_address]] IterManagerStorage iter_manager_;
  };

  RelaxedSweepSolver(Real relax, IterManagerStorage&& iter_manager,
                     std::optional<std::size_t> min_per_thread = {})
      : relax_(relax), min_per_thread_(min_per_thread),
        iter_manager_(std::forward<IterManagerStorage>(iter_manager)) {}

  explicit RelaxedSweepSolver(Real relax, std::optional<std::size_t> min_per_thread = {})
  requires(std::is_void_v<IterManager>)
      : relax_(relax), min_per_thread_(min_per_thread) {}

  template<AnyMatrix TLhs>
  Instance<TLhs> instantiate(TLhs&& lhs, const Env auto& /*env*/) const {
    return Instance<TLhs>{std::forward<TLhs>(lhs), relax_, min_per_thread_, iter_manager_};
  }

private:
  Real relax_;
  std::optional<std::size_t> min_per_thread_;
  [[no_unique_address]] thes::VoidStorage<IterManager> iter_manager_{};
};
} // namespace lineal

#endif // INCLUDE_LINEAL_LINEAR_SOLVER_ITERATIVE_STATIONARY_BROOM_HPP
