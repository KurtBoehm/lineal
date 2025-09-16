// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_LINEAR_SOLVER_DIRECT_LU_SOLVER_HPP
#define INCLUDE_LINEAL_LINEAR_SOLVER_DIRECT_LU_SOLVER_HPP

#include <cstddef>
#include <type_traits>
#include <utility>

#include "lineal/base.hpp"
#include "lineal/linear-solver/direct/lu/decomposition.hpp"
#include "lineal/linear-solver/direct/substitution.hpp"
#include "lineal/parallel.hpp"

namespace lineal {
template<typename TReal, AnyMatrix TLuMat>
struct LuSolver : public SharedDirectSolverBase, public DistributedDirectSolverBase {
  using Real = TReal;
  using LuMatrix = std::decay_t<TLuMat>;

  template<AnyMatrix TLhs>
  struct Instance {
    using Real = TReal;
    using Lhs = std::decay_t<TLhs>;
    using DistributedInfo = DistributedInfoOf<Lhs>;

    static constexpr std::size_t aux_size = 1;

    explicit Instance(TLhs&& lhs)
        : lhs_(std::forward<TLhs>(lhs)), lu_(lu_decompose<TReal, TLuMat>(lhs_)) {}

    template<SharedVector TSol, typename TAux>
    void solve(TSol& sol, const SharedVector auto& rhs, TAux&& aux, auto& env)
    requires(SharedMatrix<Lhs> && thes::star::size<TAux> >= 1)
    {
      decltype(auto) expo = env.execution_policy();
      SharedVector auto& aux_vec = thes::star::get_at<0>(std::forward<TAux>(aux));
      forward_substitute<Real>(lower(), aux_vec, rhs, expo);
      backward_substitute<Real>(upper(), sol, aux_vec, expo);
    }

    [[nodiscard]] const Lhs& lhs() const {
      return lhs_;
    }
    [[nodiscard]] const LuMatrix& lower() const {
      return lu_.first;
    }
    [[nodiscard]] const LuMatrix& upper() const {
      return lu_.second;
    }

  private:
    TLhs lhs_;
    std::pair<TLuMat, TLuMat> lu_;
  };

  LuSolver() = default;

  template<AnyMatrix TLhs>
  Instance<TLhs> instantiate(TLhs&& lhs, const Env auto& /*env*/) const {
    return Instance<TLhs>(std::forward<TLhs>(lhs));
  }
};
} // namespace lineal

#endif // INCLUDE_LINEAL_LINEAR_SOLVER_DIRECT_LU_SOLVER_HPP
