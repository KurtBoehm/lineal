// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_LINEAR_SOLVER_RUNNER_HPP
#define INCLUDE_LINEAL_LINEAR_SOLVER_RUNNER_HPP

#include <cstddef>

#include "thesauros/types.hpp"

#include "lineal/base.hpp"

namespace lineal {
template<typename TSolver, thes::AnyBoolTag TVerbose = thes::BoolTag<false>>
requires(StationaryIterativeSolver<TSolver> || NonStationaryIterativeSolver<TSolver>)
inline auto run_iterative_solver(const TSolver& solver, const AnyMatrix auto& lhs,
                                 AnyVector auto& sol, const AnyVector auto& rhs, auto aux,
                                 const auto& env, auto iter_op, TVerbose verbose = {}) {
  auto solver_instance = solver.instantiate(lhs, env);

  auto env_init = [&] {
    if constexpr (verbose) {
      return env.add_object("initialization");
    } else {
      return env.dummy();
    }
  };
  auto env_wrap = [&] {
    if constexpr (verbose) {
      return env.add_object("iterative_solve");
    } else {
      return env.dummy();
    }
  };
  auto env_iters = [&](auto& envi) {
    if constexpr (verbose) {
      return envi.add_array("iterations");
    } else {
      return envi.dummy();
    }
  };
  auto env_iter = [&](auto& envi) {
    if constexpr (verbose) {
      return envi.add_object();
    } else {
      return envi.dummy();
    }
  };

  if constexpr (NonStationaryIterativeSolver<TSolver>) {
    auto system_instance = [&] {
      decltype(auto) envi = env_init();
      return solver_instance.instantiate(sol, rhs, envi);
    }();
    auto envi = env_wrap();
    auto envii = env_iters(envi);
    for (std::size_t i = 0;; ++i) {
      decltype(auto) enviii = env_iter(envii);
      system_instance.iterate(enviii);
      if (decltype(auto) res = iter_op(i, system_instance, enviii); res.has_value()) {
        return *res;
      }
    }
  } else {
    auto envi = env_wrap();
    auto envii = env_iters(envi);
    for (std::size_t i = 1;; ++i) {
      decltype(auto) enviii = env_iter(envii);
      solver_instance.apply(sol, rhs, aux, enviii);
      if (decltype(auto) res = iter_op(i, solver_instance, enviii); res.has_value()) {
        return *res;
      }
    }
  }
}
} // namespace lineal

#endif // INCLUDE_LINEAL_LINEAR_SOLVER_RUNNER_HPP
