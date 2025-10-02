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
  using Lhs = std::decay_t<TLhs>;

  template<SharedVector TSol>
  static constexpr void forward(const TLhs& lhs, TSol&& sol, TRhs&& rhs,
                                AnyLhsHasUnitDiagonalTag auto lhs_has_unit_diagonal,
                                const auto& /*expo*/) {
    using Sol = std::decay_t<TSol>;
    using SolValue = Sol::Value;
    using SolScalar = ScalarType<SolValue>;
    using Work = WithScalarType<SolValue, Real>;
    using MatWork = WithScalarType<typename Lhs::Value, Real>;
    assert(lhs.row_num() == lhs.column_num());

    for (const auto row : lhs) {
      const auto i = row.ext_index();
      Work sum = compat::cast<Real>(rhs[i]);
      MatWork diag = compat::zero<MatWork>();

      row.iterate(
        [&](auto j, auto v)
          THES_ALWAYS_INLINE { sum -= compat::cast<Real>(v) * compat::cast<Real>(sol[j]); },
        [&](auto /*j*/, auto v) THES_ALWAYS_INLINE { diag = compat::cast<Real>(v); },
        [](auto /*j*/, auto /*v*/) THES_ALWAYS_INLINE { std::terminate(); }, valued_tag,
        unordered_tag);

      assert(diag != compat::zero<MatWork>());
      sol[i] = compat::cast<SolScalar>(
        compat::solve_tri<Real>(diag, sum, tri_lower_tag, lhs_has_unit_diagonal));
    }
  }

  template<SharedVector TSol>
  static constexpr void backward(const TLhs& lhs, TSol&& sol, TRhs&& rhs,
                                 AnyLhsHasUnitDiagonalTag auto lhs_has_unit_diagonal,
                                 const auto& /*expo*/) {
    using Sol = std::decay_t<TSol>;
    using SolValue = Sol::Value;
    using SolScalar = ScalarType<SolValue>;
    using Work = WithScalarType<SolValue, Real>;
    using MatWork = WithScalarType<typename Lhs::Value, Real>;
    assert(lhs.row_num() == lhs.column_num());

    for (const auto row : thes::reversed(lhs)) {
      const auto i = row.ext_index();
      Work sum = compat::cast<Real>(rhs[i]);
      MatWork diag = compat::zero<MatWork>();

      row.iterate([](auto /*j*/, auto /*v*/) THES_ALWAYS_INLINE { std::terminate(); },
                  [&](auto /*j*/, auto v) THES_ALWAYS_INLINE { diag = compat::cast<Real>(v); },
                  [&](auto j, auto v) THES_ALWAYS_INLINE {
                    sum -= compat::cast<Real>(v) * compat::cast<Real>(sol[j]);
                  },
                  valued_tag, unordered_tag);

      assert(diag != compat::zero<MatWork>());
      sol[i] = compat::cast<SolScalar>(
        compat::solve_tri<Real>(diag, sum, tri_upper_tag, lhs_has_unit_diagonal));
    }
  }

  template<SharedVector TSol>
  static constexpr void backward_trans(const TLhs& lhs, TSol&& sol, TRhs&& rhs,
                                       AnyLhsHasUnitDiagonalTag auto lhs_has_unit_diagonal,
                                       const auto& expo) {
    using Sol = std::decay_t<TSol>;
    using SolValue = Sol::Value;
    using SolScalar = ScalarType<SolValue>;
    using Work = WithScalarType<SolValue, Real>;
    assert(lhs.row_num() == lhs.column_num());

    // The result is not computed row by row, as usual, but column by column,
    // since the transpose of a matrix stored row by row is only available column by column.
    assign(sol, std::forward<TRhs>(rhs), expo);
    for (const auto row : thes::reversed(lhs)) {
      const auto j = row.ext_index();
      assert(row.back_column() == j);

      const Work sol_j =
        compat::solve_tri<Real>(compat::transpose(compat::cast<Real>(row.back())),
                                compat::cast<Real>(sol[j]), tri_upper_tag, lhs_has_unit_diagonal);
      sol[j] = compat::cast<SolScalar>(sol_j);

      row.iterate(
        [&](auto i, auto v) THES_ALWAYS_INLINE {
          sol[i] = compat::cast<SolScalar>(compat::cast<Real>(sol[i]) -
                                           compat::transpose(compat::cast<Real>(v)) * sol_j);
        },
        [](auto /*i*/, auto /*v*/) THES_ALWAYS_INLINE {},
        [](auto /*i*/, auto /*v*/) THES_ALWAYS_INLINE { std::terminate(); }, valued_tag,
        unordered_tag);
    }
  }
};

template<typename TReal, SharedMatrix TLhs, SharedVector TSol, SharedVector TRhs>
constexpr void forward_substitute(const TLhs& lhs, TSol&& sol, TRhs&& rhs,
                                  AnyLhsHasUnitDiagonalTag auto lhs_has_unit_diagonal,
                                  const auto& expo) {
  return SubstitutionSolver<TReal, TLhs, TRhs>::forward(
    lhs, std::forward<TSol>(sol), std::forward<TRhs>(rhs), lhs_has_unit_diagonal, expo);
}
template<typename TReal, SharedMatrix TLhs, SharedVector TSol, SharedVector TRhs>
constexpr void backward_substitute(const TLhs& lhs, TSol&& sol, TRhs&& rhs,
                                   AnyLhsHasUnitDiagonalTag auto lhs_has_unit_diagonal,
                                   const auto& expo) {
  return SubstitutionSolver<TReal, TLhs, TRhs>::backward(
    lhs, std::forward<TSol>(sol), std::forward<TRhs>(rhs), lhs_has_unit_diagonal, expo);
}
template<typename TReal, SharedMatrix TLhs, SharedVector TSol, SharedVector TRhs>
constexpr void backward_substitute_transposed(const TLhs& lhs, TSol&& sol, TRhs&& rhs,
                                              AnyLhsHasUnitDiagonalTag auto lhs_has_unit_diagonal,
                                              const auto& expo) {
  return SubstitutionSolver<TReal, TLhs, TRhs>::backward_trans(
    lhs, std::forward<TSol>(sol), std::forward<TRhs>(rhs), lhs_has_unit_diagonal, expo);
}
} // namespace lineal

#endif // INCLUDE_LINEAL_LINEAR_SOLVER_DIRECT_SUBSTITUTION_HPP
