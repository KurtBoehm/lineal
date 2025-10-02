// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_TENSOR_FIXED_COMPAT_HPP
#define INCLUDE_LINEAL_TENSOR_FIXED_COMPAT_HPP

#include <concepts>
#include <cstddef>
#include <type_traits>

#include "lineal/base/compat.hpp"
#include "lineal/base/concept.hpp"
#include "lineal/base/enum.hpp"
#include "lineal/base/type-trait.hpp"
#include "lineal/tensor/fixed/concepts.hpp"
#include "lineal/tensor/fixed/matrix.hpp"
#include "lineal/tensor/fixed/solve.hpp"
#include "lineal/tensor/fixed/vector.hpp"

namespace lineal {
template<fix::AnyMatrix TX, fix::AnyVector TY, fix::AnyVector TZ>
requires(TX::dimensions.column_num == TY::size && TX::dimensions.row_num == TZ::size)
struct compat::FmaTrait<TX, TY, TZ> {
  [[nodiscard]] static auto fmadd(const TX& x, const TY& y, const TZ& z) {
    return x * y + z;
  }
  [[nodiscard]] static auto fnmadd(const TX& x, const TY& y, const TZ& z) {
    return z - x * y;
  }
};

template<IsScalar TX, fix::AnyVector TY, fix::AnyVector TZ>
requires(TY::size == TZ::size)
struct compat::FmaTrait<TX, TY, TZ> {
  [[nodiscard]] static auto fmadd(const TX& x, const TY& y, const TZ& z) {
    return x * y + z;
  }
  [[nodiscard]] static auto fnmadd(const TX& x, const TY& y, const TZ& z) {
    return z - x * y;
  }
};

template<fix::AnyMatrix TMat>
struct compat::CompatTrait<TMat> {
  static TMat zero() {
    return TMat::zero();
  }
  static TMat identity() {
    return TMat::identity();
  }
  static TMat signaling_nan() {
    return TMat::signaling_nan();
  }

  static bool is_finite(const TMat& x) {
    return fix::is_finite(x);
  }

  template<typename TMatArg>
  requires(std::same_as<std::decay_t<TMatArg>, TMat>)
  static auto transpose(TMatArg&& mat) {
    return fix::transpose(std::forward<TMatArg>(mat));
  }
  template<fix::AnyMatrix TSol>
  static TSol cholesky_lower(const TMat& mat) {
    return fix::decompose_cholesky<TSol>(mat);
  }

  template<typename TReal>
  struct LuDecomposition {
    using Mat = WithScalarType<TMat, TReal>;
    Mat l;
    Mat u;

    constexpr const Mat& lower() const {
      return l;
    }
    constexpr const Mat& upper() const {
      return u;
    }
  };
  template<typename TReal>
  static LuDecomposition<TReal> lu_decompose(const TMat& mat) {
    using Mat = WithScalarType<TMat, TReal>;
    auto [l, u] = fix::decompose_lu<Mat, Mat>(mat);
    return {.l = std::move(l), .u = std::move(u)};
  }
};
template<fix::AnyVector TVec>
struct compat::CompatTrait<TVec> {
  using Value = TVec::Value;

  static TVec zero() {
    return TVec::zero();
  }

  template<typename TReal>
  static TReal euclidean_squared(const TVec& vec) {
    return fix::euclidean_squared<TReal>(vec);
  }
  template<typename TReal>
  static TReal max_norm(const TVec& vec) {
    return fix::max_norm<TReal>(vec);
  }
};
template<fix::AnyVector T1, fix::AnyVector T2>
struct compat::BiCompatTrait<T1, T2> {
  template<typename TReal>
  static TReal dot(const T1& x, const T2& y) {
    return fix::dot<TReal>(x, y);
  }
};

template<typename TWork, fix::AnyMatrix TLhs, fix::AnyMatrix TRhs>
requires(TLhs::dimensions.is_square() && TLhs::dimensions == TRhs::dimensions)
struct compat::SolveTrait<TWork, TLhs, TRhs> {
  static constexpr std::size_t dim = TLhs::dimensions.row_num;
  using Sol = fix::DenseMatrix<TWork, dim, dim>;

  static constexpr Sol solve(const TLhs& lhs, const TRhs& rhs) {
    return fix::solve<Sol>(lhs, rhs);
  }
  static constexpr Sol solve_tri(const TLhs& lhs, const TRhs& rhs, AnyTriangularKindTag auto kind,
                                 AnyLhsHasUnitDiagonalTag auto lhs_has_unit_diagonal) {
    Sol out{rhs};
    fix::solve_tri(lhs, out, kind, lhs_has_unit_diagonal);
    return out;
  }
  static constexpr Sol solve_right(const TLhs& lhs, const TRhs& rhs) {
    return fix::solve_right<Sol>(lhs, rhs);
  }
  static constexpr Sol solve_right_tri(const TLhs& lhs, const TRhs& rhs,
                                       AnyTriangularKindTag auto kind,
                                       AnyLhsHasUnitDiagonalTag auto lhs_has_unit_diagonal) {
    Sol out{rhs};
    fix::solve_right_tri(lhs, out, kind, lhs_has_unit_diagonal);
    return out;
  }
};
template<typename TWork, fix::AnyMatrix TLhs, fix::AnyVector TRhs>
requires(TLhs::dimensions.is_square() && TLhs::dimensions.row_num == TRhs::size)
struct compat::SolveTrait<TWork, TLhs, TRhs> {
  static constexpr std::size_t dim = TLhs::dimensions.row_num;
  using Sol = fix::DenseVector<TWork, dim>;

  static constexpr Sol solve(const TLhs& lhs, const TRhs& rhs) {
    return fix::solve<Sol>(lhs, rhs);
  }
  static constexpr Sol solve_tri(const TLhs& lhs, const TRhs& rhs, AnyTriangularKindTag auto kind,
                                 AnyLhsHasUnitDiagonalTag auto lhs_has_unit_diagonal) {
    Sol out{rhs};
    fix::solve_tri(lhs, out, kind, lhs_has_unit_diagonal);
    return out;
  }
};

template<typename TDst, fix::AnyMatrix TSrc>
struct compat::CastTrait<TDst, TSrc> {
  static constexpr auto dimensions = std::decay_t<TSrc>::dimensions;
  using DstMat = fix::DenseMatrix<TDst, dimensions.row_num, dimensions.column_num>;
  static DstMat cast(TSrc&& src) {
    return DstMat(fix::cast<TDst>(std::forward<TSrc>(src)));
  }
};
template<typename TDst, fix::AnyVector TSrc>
struct compat::CastTrait<TDst, TSrc> {
  using DstVec = fix::DenseVector<TDst, std::decay_t<TSrc>::size>;
  static DstVec cast(TSrc&& src) {
    return DstVec(fix::cast<TDst>(std::forward<TSrc>(src)));
  }
};
} // namespace lineal

#endif // INCLUDE_LINEAL_TENSOR_FIXED_COMPAT_HPP
