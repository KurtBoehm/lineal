// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef TEST_AUX_ITERATIVE_HPP
#define TEST_AUX_ITERATIVE_HPP

#include <cstddef>
#include <optional>
#include <tuple>
#include <type_traits>

#include "thesauros/test.hpp"
#include "thesauros/types.hpp"

#include "lineal/base.hpp"
#include "lineal/component-wise.hpp"
#include "lineal/linear-solver.hpp"
#include "lineal/tensor.hpp"

namespace lineal::test {
template<IsScalar TReal, thes::AnyBoolTag TVerbose = thes::BoolTag<false>>
inline auto cg(const auto& solver, const AnyMatrix auto& lhs, AnyVector auto& sol,
               const AnyVector auto& rhs, const auto& env, const std::size_t iter_num,
               TVerbose verbose = {}) {
  return run_iterative_solver(
    solver, lhs, sol, rhs, std::tuple{}, env,
    [&](auto iter, auto& inst, auto& envi) {
      const auto res = inst.residual_norm();
      const auto rho = inst.rho();
      THES_ALWAYS_ASSERT(std::isfinite(res));

      if constexpr (verbose) {
        envi.log("iteration", iter);
        envi.log("residual", res);
        envi.log("rho", rho);
      }

      using Out = std::optional<decltype(res)>;
      return (iter > iter_num || rho == 0) ? Out{res} : Out{};
    },
    verbose);
}

template<IsScalar TReal, AnyVector TSol, thes::AnyBoolTag TVerbose = thes::BoolTag<false>>
inline TReal gs(const auto& solver, const AnyMatrix auto& lhs, TSol& sol, const AnyVector auto& rhs,
                const auto& env, const std::size_t iter_num, TVerbose verbose = {}) {
  auto aux = create_numa_undef_like<TSol>(sol, env);
  return run_iterative_solver(
    solver, lhs, sol, rhs, std::tie(aux), env,
    [&](auto iter, auto& /*inst*/, auto& envi) {
      const auto res = euclidean_norm<TReal>(lhs * sol - rhs, envi.execution_policy());

      if constexpr (verbose) {
        envi.log("iteration", iter);
        envi.log("residual", res);
      }

      using Out = std::optional<decltype(res)>;
      return (iter > iter_num) ? Out{res} : Out{};
    },
    verbose);
}

template<IsScalar TReal, AnyVector TCoarseVec, AnyMatrix TMat>
inline TReal direct(const TMat& mat, const auto& wright_instance, auto& env) {
  using CoarseValue = std::decay_t<TCoarseVec>::Value;
  const auto& expo = env.execution_policy();

  auto hierarchy = wright_instance.create(mat, env);
  const auto& lhs = hierarchy.coarsest_lhs();
  auto& sol_inst = hierarchy.coarsest_solver_instance();

  auto sol_ref = create_constant_like<TCoarseVec>(lhs, compat::zero<CoarseValue>(), env);
  const auto rhs = create_from<TCoarseVec>(lhs * sol_ref, expo);

  auto sol = create_numa_undef_like<TCoarseVec>(rhs, env);
  auto aux = create_numa_undef_like<TCoarseVec>(rhs, env);
  sol_inst.solve(sol, rhs, std::tie(aux), env);
  return max_norm<TReal>(sol - sol_ref, expo);
}

template<typename TDefs, SorVariant tSorVariant, thes::AnyBoolTag TVerbose = thes::BoolTag<false>>
inline void suite(const auto& lhs, auto& sol, const auto& rhs, const auto& wright_instance,
                  auto& env, auto op, TVerbose verbose = {}) {
  using Real = TDefs::Real;
  using VecValue = TDefs::VecValue;
  using IterMan = FixedIterationManager;
  const bool is_symmetric = lhs.is_symmetric() == IsSymmetric{true};

  const auto& expo = env.execution_policy();
  auto zero = [&] { assign(sol, constant_like(sol, compat::zero<VecValue>()), expo); };
  auto sol_op = [&](auto res, const auto& envi) { op(res, sol, envi); };

  auto rere_controller = [](auto iter, auto /*res*/, auto& /*env*/) {
    return iter > 0 && iter % 8 == 0;
  };

  if (is_symmetric) {
    auto envi = env.add_object("cg");
    zero();
    ConjugateGradientsSolver solver{thes::type_tag<TDefs>, rere_controller};
    sol_op(cg<Real>(solver, lhs, sol, rhs, envi, 128, verbose), envi);
  }

  {
    auto envi = env.add_object("bicgstab");
    zero();
    BiCgStabSolver solver{thes::type_tag<TDefs>, rere_controller};
    sol_op(cg<Real>(solver, lhs, sol, rhs, envi, 128, verbose), envi);
  }

  {
    auto envi = env.add_object("sor");
    zero();
    SorSolver<Real, void, forward_tag, tSorVariant> solver{1.0};
    sol_op(gs<Real>(solver, lhs, sol, rhs, envi, 128, verbose), envi);
  }

  {
    auto envi = env.add_object("ssor1");
    zero();
    SsorSolver<Real, void, tSorVariant> solver{1.0};
    sol_op(gs<Real>(solver, lhs, sol, rhs, envi, 128, verbose), envi);
  }

  {
    auto envi = env.add_object("sor5");
    zero();
    SorSolver<Real, IterMan, forward_tag, tSorVariant> solver{1.0, IterMan{5}};
    sol_op(gs<Real>(solver, lhs, sol, rhs, envi, 128, verbose), envi);
  }

  {
    auto envi = env.add_object("ssor5");
    zero();
    SsorSolver<Real, IterMan, tSorVariant> solver{1.0, IterMan{5}};
    sol_op(gs<Real>(solver, lhs, sol, rhs, envi, 128, verbose), envi);
  }

  if (is_symmetric) {
    auto envi = env.add_object("pcg_ssor1");
    zero();
    using Precon = SsorSolver<Real, IterMan, tSorVariant>;
    ConjugateGradientsSolver solver{
      thes::type_tag<TDefs>,
      rere_controller,
      Precon{1.0, IterMan{1}},
    };
    sol_op(cg<Real>(solver, lhs, sol, rhs, envi, 128, verbose), envi);
  }

  {
    auto envi = env.add_object("pbicgstab_ssor1");
    zero();
    using Precon = SsorSolver<Real, IterMan, tSorVariant>;
    BiCgStabSolver solver{
      thes::type_tag<TDefs>,
      rere_controller,
      Precon{1.0, IterMan{1}},
    };
    sol_op(cg<Real>(solver, lhs, sol, rhs, envi, 128, verbose), envi);
  }

  if (is_symmetric) {
    auto envi = env.add_object("pcg_amg");
    zero();
    MultiGridSolver precon{1.0, DefaultCycle{CycleKind::REGULAR, 1}, wright_instance};
    ConjugateGradientsSolver solver{thes::type_tag<TDefs>, rere_controller, precon};
    sol_op(cg<Real>(solver, lhs, sol, rhs, envi, 32, verbose), envi);
  }

  {
    auto envi = env.add_object("pbicgstab_amg");
    zero();
    MultiGridSolver precon{1.0, DefaultCycle{CycleKind::REGULAR, 1}, wright_instance};
    BiCgStabSolver solver{thes::type_tag<TDefs>, rere_controller, precon};
    sol_op(cg<Real>(solver, lhs, sol, rhs, envi, 32, verbose), envi);
  }
}

template<typename TDefs, thes::AnyBoolTag TVerbose = thes::BoolTag<false>>
inline void suite(const auto& lhs, auto& sol, const auto& rhs, const auto& wright_instance,
                  auto& env, auto op, TVerbose verbose = {}) {
  {
    auto envi = env.add_object("sor_regular");
    suite<TDefs, SorVariant::regular>(lhs, sol, rhs, wright_instance, envi, op, verbose);
  }
  {
    auto envi = env.add_object("sor_ultra");
    suite<TDefs, SorVariant::ultra>(lhs, sol, rhs, wright_instance, envi, op, verbose);
  }
}
} // namespace lineal::test

#endif // TEST_AUX_ITERATIVE_HPP
