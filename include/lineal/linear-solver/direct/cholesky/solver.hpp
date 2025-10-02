// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_LINEAR_SOLVER_DIRECT_CHOLESKY_SOLVER_HPP
#define INCLUDE_LINEAL_LINEAR_SOLVER_DIRECT_CHOLESKY_SOLVER_HPP

#include <cstddef>
#include <type_traits>

#include "thesauros/static-ranges.hpp"

#include "lineal/base.hpp"
#include "lineal/linear-solver/direct/cholesky/decomposition.hpp"
#include "lineal/linear-solver/direct/substitution.hpp"
#include "lineal/parallel.hpp"

namespace lineal {
template<typename TReal, AnyMatrix TLowerMat>
struct CholeskySolver : public SharedDirectSolverBase, public DistributedDirectSolverBase {
  using Real = TReal;
  using LowerMatrix = std::decay_t<TLowerMat>;

  template<AnyMatrix TLhs>
  struct Instance {
    using Real = TReal;
    using Lhs = std::decay_t<TLhs>;
    using DistributedInfo = DistributedInfoOf<Lhs>;

    static constexpr std::size_t aux_size = 1;

    explicit Instance(TLhs&& lhs)
        : lhs_(std::forward<TLhs>(lhs)), lower_(cholesky_decompose<TReal, TLowerMat>(lhs_)) {}

    const LowerMatrix& lower() const {
      return lower_;
    }

    template<SharedVector TSol, typename TAux>
    void solve(TSol& sol, const SharedVector auto& rhs, TAux&& aux, auto& env)
    requires(SharedMatrix<Lhs> && thes::star::size<TAux> >= 1)
    {
      decltype(auto) expo = env.execution_policy();
      SharedVector auto& aux_vec = thes::star::get_at<0>(std::forward<TAux>(aux));
      forward_substitute<Real>(lower_, aux_vec, rhs, lhs_has_unit_diagonal_tag<false>, expo);
      backward_substitute_transposed<Real>(lower_, sol, aux_vec, lhs_has_unit_diagonal_tag<false>,
                                           expo);
    }

    [[nodiscard]] const Lhs& lhs() const {
      return lhs_;
    }

  private:
    TLhs lhs_;
    TLowerMat lower_;
  };

  CholeskySolver() = default;

  template<AnyMatrix TLhs>
  Instance<TLhs> instantiate(TLhs&& lhs, const Env auto& /*env*/) const {
    return Instance<TLhs>(std::forward<TLhs>(lhs));
  }
};
} // namespace lineal

#endif // INCLUDE_LINEAL_LINEAR_SOLVER_DIRECT_CHOLESKY_SOLVER_HPP
