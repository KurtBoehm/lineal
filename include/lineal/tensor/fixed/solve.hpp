// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_TENSOR_FIXED_SOLVE_HPP
#define INCLUDE_LINEAL_TENSOR_FIXED_SOLVE_HPP

#include <cstddef>
#include <cstdlib>
#include <type_traits>
#include <utility>

#include "thesauros/macropolis.hpp"
#include "thesauros/static-ranges.hpp"
#include "thesauros/types.hpp"

#include "lineal/base/compat/operation.hpp"
#include "lineal/base/enum.hpp"
#include "lineal/tensor/fixed/concepts.hpp"
#include "lineal/tensor/fixed/matrix.hpp"

namespace lineal::fix {
// Perform backward substitution, where `rhs` is modified to contain the solution at termination
// If `lhs_has_unit_diagonal`, the solution is not divided by the diagonal element
template<AnyMatrix TLhs, AnyMatrix TRhs>
requires(TLhs::dimensions.is_square() && TLhs::dimensions == std::decay_t<TRhs>::dimensions)
THES_ALWAYS_INLINE constexpr void solve_tri(const TLhs& lhs, TRhs&& rhs,
                                            AnyTriangularKindTag auto kind,
                                            AnyLhsHasUnitDiagonalTag auto lhs_has_unit_diagonal) {
  using Rhs = std::decay_t<TRhs>;
  using Real = Rhs::Value;
  constexpr std::size_t size = Rhs::dimensions.row_num;

  if constexpr (size < 1) {
    return;
  }

  // backward elimination
  const auto irange = [&] {
    if constexpr (kind == tri_lower) {
      return thes::star::iota<0, size>;
    } else {
      return thes::star::iota<0, size> | thes::star::reversed;
    }
  }();
  irange | thes::star::for_each([&](auto i) {
    const auto jrange = [&] {
      if constexpr (kind == tri_lower) {
        return thes::star::iota<0, i>;
      } else {
        return thes::star::iota<i + 1, size>;
      }
    }();
    jrange | thes::star::for_each([&](auto j) {
      const auto factor = Real(lhs[i, j]);
      thes::star::iota<0, size> |
        thes::star::for_each([&](auto k) { rhs[i, k] -= factor * rhs[j, k]; });
    });
    if constexpr (lhs_has_unit_diagonal == lineal::lhs_has_unit_diagonal<false>) {
      const auto factor = Real(lhs[i, i]);
      thes::star::iota<0, size> | thes::star::for_each([&](auto k) { rhs[i, k] /= factor; });
    }
  });
}
template<AnyMatrix TLhs, AnyMatrix TRhs>
requires(TLhs::dimensions.is_square() && TLhs::dimensions == TRhs::dimensions)
THES_ALWAYS_INLINE constexpr void
solve_right_tri(const TLhs& lhs, TRhs& rhs, AnyTriangularKindTag auto kind,
                AnyLhsHasUnitDiagonalTag auto lhs_has_unit_diagonal) {
  solve_tri(transpose(lhs), transpose(rhs), thes::auto_tag<!kind>, lhs_has_unit_diagonal);
}

template<AnyMatrix TLhs, AnyVector TRhs>
requires(TLhs::dimensions.is_square() && TLhs::dimensions.row_num == TRhs::size)
THES_ALWAYS_INLINE constexpr void solve_tri(const TLhs& lhs, TRhs& rhs,
                                            AnyTriangularKindTag auto kind,
                                            AnyLhsHasUnitDiagonalTag auto lhs_has_unit_diagonal) {
  using Real = TRhs::Value;
  constexpr std::size_t size = TRhs::size;

  const auto irange = [&] {
    if constexpr (kind == tri_lower) {
      return thes::star::iota<0, size>;
    } else {
      return thes::star::iota<0, size> | thes::star::reversed;
    }
  }();
  irange | thes::star::for_each([&](auto i) {
    Real tmp = rhs[i];
    const auto jrange = [&] {
      if constexpr (kind == tri_lower) {
        return thes::star::iota<0, i>;
      } else {
        return thes::star::iota<i + 1, size>;
      }
    }();
    jrange | thes::star::for_each([&](auto j) { tmp -= Real(lhs[i, j]) * rhs[j]; });
    if (lhs_has_unit_diagonal == lineal::lhs_has_unit_diagonal<false>) {
      tmp /= Real(lhs[i, i]);
    }
    rhs[i] = tmp;
  });
}

template<AnyMatrix TLhs, AnyVector TRhs>
requires(TLhs::dimensions.is_square() && TLhs::dimensions.row_num == TRhs::size)
THES_ALWAYS_INLINE constexpr void gauss_elimination(TLhs& lhs, TRhs& rhs) {
  using LVal = TLhs::Value;
  using RVal = TRhs::Value;
  constexpr std::size_t size = TRhs::size;

  if constexpr (size < 2) {
    return;
  }

  thes::star::iota<0, size - 1> | thes::star::for_each([&](auto k) {
    // partial pivoting
    std::size_t imax = k;
    LVal lmax = std::abs(lhs[k, k]);
    thes::star::iota<k + 1, size> | thes::star::for_each([&](auto i) {
      if (const auto v = std::abs(lhs[i, k]); v > lmax) {
        imax = i;
        lmax = v;
      }
    });

    // if the current row is not the pivot row, swap row with pivot row
    if (imax != k.value) {
      thes::star::iota<k, size> |
        thes::star::for_each([&](auto j) { std::swap(lhs[imax, j], lhs[k, j]); });
      std::swap(rhs[imax], rhs[k]);
    }

    // eliminate the rows below the pivot element
    thes::star::iota<k + 1, size> | thes::star::for_each([&](auto i) {
      const auto factor = lhs[i, k] / lhs[k, k];
      if (factor != 0) {
        thes::star::iota<k, size> |
          thes::star::for_each([&](auto j) { lhs[i, j] -= factor * lhs[k, j]; });
      }
      rhs[i] -= RVal(factor) * rhs[k];
    });
  });
}

template<AnyMatrix TLhs, AnyMatrix TRhs>
requires(std::decay_t<TLhs>::dimensions.is_square() &&
         std::decay_t<TLhs>::dimensions == std::decay_t<TRhs>::dimensions)
THES_ALWAYS_INLINE constexpr void gauss_elimination_matrix(TLhs&& lhs, TRhs&& rhs) {
  using LVal = std::decay_t<TLhs>::Value;
  using RVal = std::decay_t<TRhs>::Value;
  constexpr std::size_t size = std::decay_t<TRhs>::dimensions.row_num;

  if constexpr (size < 1) {
    return;
  }

  // forward elinination
  thes::star::iota<0, size - 1> | thes::star::for_each([&](auto k) {
    // partial pivoting
    std::size_t imax = k;
    LVal lmax = std::abs(lhs[k, k]);
    thes::star::iota<k + 1, size> | thes::star::for_each([&](auto i) {
      if (const auto v = std::abs(lhs[i, k]); v > lmax) {
        imax = i;
        lmax = v;
      }
    });

    // if the current row is not the pivot row, swap row with pivot row
    // and scale the elements so that the a[k, k] = 1 after swapping
    {
      const LVal factor = lhs[imax, k];
      thes::star::iota<k, size> | thes::star::for_each([&](auto j) {
        std::swap(lhs[imax, j], lhs[k, j]);
        lhs[k, j] /= factor;
      });

      const auto rfactor = RVal(factor);
      thes::star::iota<0, size> | thes::star::for_each([&](auto j) {
        std::swap(rhs[imax, j], rhs[k, j]);
        rhs[k, j] /= rfactor;
      });
    }

    // eliminate the rows below the pivot element
    thes::star::iota<k + 1, size> | thes::star::for_each([&](auto i) {
      auto factor = lhs[i, k];
      thes::star::iota<k, size> |
        thes::star::for_each([&](auto j) { lhs[i, j] -= factor * lhs[k, j]; });
      thes::star::iota<0, size> |
        thes::star::for_each([&](auto j) { rhs[i, j] -= RVal(factor) * rhs[k, j]; });
    });
  });

  // scale last row
  {
    constexpr thes::IndexTag<size - 1> last{};
    const auto factor = RVal(lhs[last, last]);
    lhs[last, last] = 1;
    thes::star::iota<0, size> | thes::star::for_each([&](auto i) { rhs[last, i] /= factor; });
  }

  solve_tri(lhs, rhs, tri_upper_tag, lhs_has_unit_diagonal_tag<true>);
}

template<AnyMatrix TSol, AnyMatrix TMat>
requires(TMat::dimensions.row_num == 2 && TMat::dimensions.column_num == 2)
THES_ALWAYS_INLINE constexpr auto invert_analytic(const TMat& mat) {
  using Real = TSol::Value;

  const auto a = Real(mat[thes::index_tag<0>, thes::index_tag<0>]);
  const auto b = Real(mat[thes::index_tag<0>, thes::index_tag<1>]);
  const auto c = Real(mat[thes::index_tag<1>, thes::index_tag<0>]);
  const auto d = Real(mat[thes::index_tag<1>, thes::index_tag<1>]);
  const auto det = a * d - b * c;

  return std::pair{TSol{{d, -b}, {-c, a}}, det};
}
template<AnyMatrix TSol, AnyMatrix TMat>
requires(TMat::dimensions.row_num == 3 && TMat::dimensions.column_num == 3)
THES_ALWAYS_INLINE constexpr auto invert_analytic(const TMat& mat) {
  using Real = TSol::Value;

  const auto a = Real(mat[thes::index_tag<0>, thes::index_tag<0>]);
  const auto b = Real(mat[thes::index_tag<0>, thes::index_tag<1>]);
  const auto c = Real(mat[thes::index_tag<0>, thes::index_tag<2>]);
  const auto d = Real(mat[thes::index_tag<1>, thes::index_tag<0>]);
  const auto e = Real(mat[thes::index_tag<1>, thes::index_tag<1>]);
  const auto f = Real(mat[thes::index_tag<1>, thes::index_tag<2>]);
  const auto g = Real(mat[thes::index_tag<2>, thes::index_tag<0>]);
  const auto h = Real(mat[thes::index_tag<2>, thes::index_tag<1>]);
  const auto i = Real(mat[thes::index_tag<2>, thes::index_tag<2>]);

  const auto inv00 = e * i - f * h;
  const auto inv10 = f * g - d * i;
  const auto inv20 = d * h - e * g;

  const auto inv01 = c * h - b * i;
  const auto inv11 = a * i - c * g;
  const auto inv21 = b * g - a * h;

  const auto inv02 = b * f - c * e;
  const auto inv12 = c * d - a * f;
  const auto inv22 = a * e - b * d;

  const auto det = a * inv00 + b * inv10 + c * inv20;

  return std::pair{TSol{{inv00, inv01, inv02}, {inv10, inv11, inv12}, {inv20, inv21, inv22}}, det};
}

template<AnyMatrix TSol, AnyMatrix TMat>
requires(TSol::dimensions.is_square() && TSol::dimensions == TMat::dimensions)
THES_ALWAYS_INLINE constexpr TSol decompose_cholesky(const TMat& mat) {
  using Real = TSol::Value;
  constexpr std::size_t size = TMat::dimensions.row_num;

  if constexpr (size < 2) {
    return TSol{mat};
  }

  using Sol = TupleTriangularMatrix<Real, size, tri_lower>;
  Sol lower{};
  thes::star::iota<0, size> | thes::star::for_each([&](auto i) {
    thes::star::iota<0, i> | thes::star::for_each([&](auto j) {
      Real sum = mat[i, j];
      thes::star::iota<0, j> |
        thes::star::for_each([&](auto k) { sum -= lower[i, k] * lower[j, k]; });
      lower[i, j] = sum / lower[j, j];
    });

    Real sum = mat[i, i];
    thes::star::iota<0, i> |
      thes::star::for_each([&](auto k) { sum -= lower[i, k] * lower[i, k]; });
    lower[i, i] = compat::sqrt(sum);
  });

  return TSol{std::move(lower)};
}

// An LU decomposition **without pivoting**
template<AnyMatrix TLower, AnyMatrix TUpper, AnyMatrix TMat>
requires(TMat::dimensions.is_square() && TMat::dimensions == TLower::dimensions &&
         TMat::dimensions == TUpper::dimensions)
THES_ALWAYS_INLINE constexpr std::pair<TLower, TUpper> decompose_lu(const TMat& mat) {
  using namespace thes::literals;
  using LowerVal = TLower::Value;
  using UpperVal = TUpper::Value;
  using Real = thes::Union<LowerVal, UpperVal>;
  constexpr std::size_t size = TMat::dimensions.row_num;

  using Lower = TupleTriangularMatrix<LowerVal, size, tri_lower>;
  using Upper = TupleTriangularMatrix<UpperVal, size, TriangularKind::upper>;
  Lower lower = Lower::identity();
  Upper upper = Upper::zero();
  thes::star::iota<0, size> | thes::star::for_each([&](auto i) {
    // Compute L entries
    thes::star::iota<0, i> | thes::star::for_each([&](auto j) {
      Real sum = mat[i, j];
      thes::star::iota<0, j> |
        thes::star::for_each([&](auto k) { sum -= lower[i, k] * upper[k, j]; });
      lower[i, j] = sum / upper[j, j];
    });
    thes::star::iota<i, size> | thes::star::for_each([&](auto j) {
      Real sum = mat[i, j];
      thes::star::iota<0, i> |
        thes::star::for_each([&](auto k) { sum -= lower[i, k] * upper[k, j]; });
      upper[i, j] = sum;
    });
  });
  return std::make_pair(TLower(std::move(lower)), TUpper(std::move(upper)));
}

template<AnyVector TSol, AnyMatrix TLhs, AnyVector TRhs>
requires(TLhs::dimensions.is_square() && TLhs::dimensions.row_num == TSol::size &&
         TSol::size == TRhs::size)
THES_ALWAYS_INLINE constexpr TSol solve(const TLhs& lhs, const TRhs& rhs) {
  using namespace thes::literals;
  using Real = TSol::Value;

  constexpr std::size_t size = TRhs::size;
  if constexpr (size == 1) {
    return TSol{rhs["0"_it] / Real(lhs["0"_it, "0"_it])};
  }
  if constexpr (size == 2 || size == 3) {
    const auto [sol, det] = invert_analytic<DenseMatrix<Real, size, size>>(lhs);
    return TSol{sol * rhs / det};
  }

  TLhs lhsc{lhs};
  TSol rhsc{rhs};
  gauss_elimination(lhsc, rhsc);
  solve_tri(lhsc, rhsc, tri_upper_tag, lhs_has_unit_diagonal_tag<false>);
  return rhsc;
}

template<AnyMatrix TSol, AnyMatrix TLhs, AnyMatrix TRhs>
requires(TSol::dimensions.is_square() && TSol::dimensions == TLhs::dimensions &&
         TSol::dimensions == TRhs::dimensions)
THES_ALWAYS_INLINE constexpr TSol solve(const TLhs& lhs, const TRhs& rhs) {
  using namespace thes::literals;
  using Real = TSol::Value;
  constexpr std::size_t size = TSol::dimensions.row_num;

  if constexpr (size == 1) {
    return TSol{rhs["0"_it, "0"_it] / Real(lhs["0"_it, "0"_it])};
  }
  if constexpr (size == 2 || size == 3) {
    const auto [sol, det] = invert_analytic<TSol>(lhs);
    return TSol{sol * rhs / det};
  }

  TLhs lhsc{lhs};
  TSol sol{rhs};
  gauss_elimination_matrix(lhsc, sol);
  return sol;
}

// Solve x·lhs = rhs
template<AnyMatrix TSol, AnyMatrix TLhs, AnyMatrix TRhs>
requires(TSol::dimensions.is_square() && TSol::dimensions == TLhs::dimensions &&
         TSol::dimensions == TRhs::dimensions)
THES_ALWAYS_INLINE constexpr TSol solve_right(const TLhs& lhs, const TRhs& rhs) {
  using namespace thes::literals;
  using Real = TSol::Value;
  constexpr std::size_t size = TSol::dimensions.row_num;

  if constexpr (size == 1) {
    return TSol{rhs["0"_it, "0"_it] / Real(lhs["0"_it, "0"_it])};
  }
  if constexpr (size == 2 || size == 3) {
    // x·lhs = rhs iff x = rhs·lhs⁻¹
    const auto [sol, det] = invert_analytic<TSol>(lhs);
    return TSol{rhs * sol / det};
  }

  // x·lhs = rhs is equivalent to lhs^T·x^T = rhs^T
  TLhs lhsc{lhs};
  TSol sol{rhs};
  gauss_elimination_matrix(transpose(lhsc), transpose(sol));
  return sol;
}

template<AnyMatrix TSol, AnyMatrix TMat>
requires(TMat::dimensions.is_square())
THES_ALWAYS_INLINE constexpr TSol invert(const TMat& mat) {
  using namespace thes::literals;
  using Real = TSol::Value;
  constexpr std::size_t size = TMat::dimensions.row_num;

  if constexpr (size == 1) {
    return TSol{{Real(1) / Real(mat["0"_it, "0"_it])}};
  }
  if constexpr (size == 2 || size == 3) {
    const auto [sol, det] = invert_analytic<TSol>(mat);
    return TSol(sol / det);
  }

  TMat lhsc{mat};
  TSol sol = TSol::identity();
  gauss_elimination_matrix(lhsc, sol);
  return sol;
}
} // namespace lineal::fix

#endif // INCLUDE_LINEAL_TENSOR_FIXED_SOLVE_HPP
