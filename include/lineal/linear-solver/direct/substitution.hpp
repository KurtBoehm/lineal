// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_LINEAR_SOLVER_DIRECT_SUBSTITUTION_HPP
#define INCLUDE_LINEAL_LINEAR_SOLVER_DIRECT_SUBSTITUTION_HPP

#include <cassert>
#include <exception>
#include <type_traits>

#include "thesauros/macropolis.hpp"
#include "thesauros/ranges.hpp"

#include "lineal/base.hpp"
#include "lineal/component-wise/sink/assignment.hpp"

namespace lineal {
template<typename TReal, SharedMatrix TLhs, SharedVector TRhs>
struct SubstitutionSolver {
  using Real = TReal;

  template<SharedVector TSol>
  static constexpr void forward(const TLhs& lhs, TSol&& sol, TRhs&& rhs, const auto& /*expo*/) {
    using SolValue = std::decay_t<TSol>::Value;
    assert(lhs.row_num() == lhs.column_num());

    for (const auto row : lhs) {
      const auto i = row.ext_index();
      Real sum = Real(rhs[i]);
      Real diag = 0;

      row.iterate([&](auto j, auto v) THES_ALWAYS_INLINE { sum -= Real(v) * Real(sol[j]); },
                  [&](auto /*j*/, auto v) THES_ALWAYS_INLINE { diag = Real(v); },
                  [](auto /*j*/, auto /*v*/) THES_ALWAYS_INLINE { std::terminate(); }, valued_tag,
                  unordered_tag);

      assert(diag != 0);
      sol[i] = SolValue(sum / diag);
    }
  }

  template<SharedVector TSol>
  static constexpr void backward(const TLhs& lhs, TSol&& sol, TRhs&& rhs, const auto& /*expo*/) {
    using SolValue = std::decay_t<TSol>::Value;
    assert(lhs.row_num() == lhs.column_num());

    for (const auto row : thes::reversed(lhs)) {
      const auto i = row.ext_index();
      Real sum = Real(rhs[i]);
      Real diag = 0;

      row.iterate([](auto /*j*/, auto /*v*/) THES_ALWAYS_INLINE { std::terminate(); },
                  [&](auto /*j*/, auto v) THES_ALWAYS_INLINE { diag = Real(v); },
                  [&](auto j, auto v) THES_ALWAYS_INLINE { sum -= Real(v) * Real(sol[j]); },
                  valued_tag, unordered_tag);

      assert(diag != 0);
      sol[i] = SolValue(sum / diag);
    }
  }

  template<SharedVector TSol>
  static constexpr void backward_trans(const TLhs& lhs, TSol&& sol, TRhs&& rhs, const auto& expo) {
    using SolValue = std::decay_t<TSol>::Value;
    assert(lhs.row_num() == lhs.column_num());

    // The result is not computed row by row, as usual, but column by column,
    // since the transpose of a matrix stored row by row is only available column by column.
    assign(sol, std::forward<TRhs>(rhs), expo);
    for (const auto row : thes::reversed(lhs)) {
      const auto j = row.ext_index();
      assert(row.back_column() == j);

      const Real sol_j = Real(sol[j]) / Real(row.back());
      sol[j] = SolValue(sol_j);

      row.iterate([&](auto i, auto v)
                    THES_ALWAYS_INLINE { sol[i] = SolValue(Real(sol[i]) - Real(v) * sol_j); },
                  [](auto /*i*/, auto /*v*/) THES_ALWAYS_INLINE {},
                  [](auto /*i*/, auto /*v*/) THES_ALWAYS_INLINE { std::terminate(); }, valued_tag,
                  unordered_tag);
    }
  }
};

template<typename TReal, SharedMatrix TLhs, SharedVector TSol, SharedVector TRhs>
constexpr void forward_substitute(const TLhs& lhs, TSol&& sol, TRhs&& rhs, const auto& expo) {
  return SubstitutionSolver<TReal, TLhs, TRhs>::forward(lhs, std::forward<TSol>(sol),
                                                        std::forward<TRhs>(rhs), expo);
}
template<typename TReal, SharedMatrix TLhs, SharedVector TSol, SharedVector TRhs>
constexpr void backward_substitute(const TLhs& lhs, TSol&& sol, TRhs&& rhs, const auto& expo) {
  return SubstitutionSolver<TReal, TLhs, TRhs>::backward(lhs, std::forward<TSol>(sol),
                                                         std::forward<TRhs>(rhs), expo);
}
template<typename TReal, SharedMatrix TLhs, SharedVector TSol, SharedVector TRhs>
constexpr void backward_substitute_transposed(const TLhs& lhs, TSol&& sol, TRhs&& rhs,
                                              const auto& expo) {
  return SubstitutionSolver<TReal, TLhs, TRhs>::backward_trans(lhs, std::forward<TSol>(sol),
                                                               std::forward<TRhs>(rhs), expo);
}
} // namespace lineal

#endif // INCLUDE_LINEAL_LINEAR_SOLVER_DIRECT_SUBSTITUTION_HPP
