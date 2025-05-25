// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_LINEAR_SOLVER_CHOLESKY_SUBSTITUTION_HPP
#define INCLUDE_LINEAL_LINEAR_SOLVER_CHOLESKY_SUBSTITUTION_HPP

#include <cassert>
#include <exception>

#include "thesauros/macropolis.hpp"
#include "thesauros/ranges.hpp"

#include "lineal/base.hpp"
#include "lineal/component-wise/sink/assignment.hpp"

namespace lineal {
template<typename TReal, SharedMatrix TLhs, SharedVector TRhs>
struct SubstitutionSolver {
  using Real = TReal;

  static constexpr void forward(const TLhs& lhs, SharedVector auto&& sol, TRhs&& rhs,
                                const auto& /*expo*/) {
    assert(lhs.row_num() == lhs.column_num());
    for (const auto row : lhs) {
      const auto i = row.ext_index();
      Real sum = rhs[i];
      Real diag = 0;

      row.iterate([&](auto j, Real v) THES_ALWAYS_INLINE { sum -= v * sol[j]; },
                  [&](auto /*j*/, Real v) THES_ALWAYS_INLINE { diag = v; },
                  [](auto /*j*/, Real /*v*/) THES_ALWAYS_INLINE { std::terminate(); }, valued_tag,
                  unordered_tag);

      assert(diag != 0);
      sol[i] = sum / diag;
    }
  }

  static constexpr void backward_trans(const TLhs& lhs, SharedVector auto&& sol, TRhs&& rhs,
                                       const auto& expo) {
    assert(lhs.row_num() == lhs.column_num());
    // The result is not computed row by row, as usual, but column by column,
    // since the transpose of a matrix stored row by row is only available column by column.
    assign(sol, std::forward<TRhs>(rhs), expo);
    for (const auto row : thes::reversed(lhs)) {
      const auto j = row.ext_index();
      assert(row.back_column() == j);

      const Real sol_j = Real(sol[j]) / Real(row.back());
      sol[j] = sol_j;

      row.iterate([&](auto i, Real v) THES_ALWAYS_INLINE { sol[i] = sol[i] - v * sol_j; },
                  [](auto /*i*/, Real /*v*/) THES_ALWAYS_INLINE {},
                  [](auto /*i*/, Real /*v*/) THES_ALWAYS_INLINE { std::terminate(); }, valued_tag,
                  unordered_tag);
    }
  }
};

template<typename TReal, SharedMatrix TLhs, SharedVector TSol, SharedVector TRhs>
inline constexpr void forward_substitute(const TLhs& lhs, TSol&& sol, TRhs&& rhs,
                                         const auto& expo) {
  return SubstitutionSolver<TReal, TLhs, TRhs>::forward(lhs, std::forward<TSol>(sol),
                                                        std::forward<TRhs>(rhs), expo);
}
template<typename TReal, SharedMatrix TLhs, SharedVector TSol, SharedVector TRhs>
inline constexpr void backward_substitute_transposed(const TLhs& lhs, TSol&& sol, TRhs&& rhs,
                                                     const auto& expo) {
  return SubstitutionSolver<TReal, TLhs, TRhs>::backward_trans(lhs, std::forward<TSol>(sol),
                                                               std::forward<TRhs>(rhs), expo);
}
} // namespace lineal

#endif // INCLUDE_LINEAL_LINEAR_SOLVER_CHOLESKY_SUBSTITUTION_HPP
