// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_BASE_COMPAT_OPERATION_HPP
#define INCLUDE_LINEAL_BASE_COMPAT_OPERATION_HPP

#include <limits>
#include <type_traits>

#include "grex/base/defs.hpp"
#include "grex/operations-tagged.hpp"
#include "grex/operations.hpp"
#include "grex/tags.hpp"
#include "grex/types.hpp"
#include "thesauros/math/compile-time.hpp"

#include "lineal/base/enum.hpp"

namespace lineal::compat {
template<typename T>
constexpr T sqrt(T x) {
  if consteval {
    return thes::ctm::sqrt(x);
  } else {
    return grex::sqrt(x);
  }
}

template<typename T>
struct AbsTrait {
  static constexpr T abs(T x) {
    if consteval {
      return std::abs(x);
    } else {
      return grex::abs(x);
    }
  }
};
template<grex::AnyVector T>
struct AbsTrait<T> {
  static T abs(T x) {
    return grex::abs(x);
  }
};
template<typename T>
constexpr T abs(T x) {
  return AbsTrait<T>::abs(x);
}

template<typename TX, typename TY, typename TZ>
struct FmaTrait {
  [[nodiscard]] static auto fmadd(TX x, TY y, TZ z) {
    return grex::fmadd(x, y, z);
  }
  [[nodiscard]] static auto fnmadd(TX x, TY y, TZ z) {
    return grex::fnmadd(x, y, z);
  }
};
template<typename TX, typename TY, typename TZ>
inline auto fmadd(const TX& x, const TY& y, const TZ& z) {
  return FmaTrait<TX, TY, TZ>::fmadd(x, y, z);
}
template<typename TX, typename TY, typename TZ>
inline auto fnmadd(const TX& x, const TY& y, const TZ& z) {
  return FmaTrait<TX, TY, TZ>::fnmadd(x, y, z);
}

template<typename T>
struct CompatTrait {
  static T zero() {
    return 0;
  }
  static T signaling_nan() {
    return std::numeric_limits<T>::signaling_NaN();
  }
  static T identity() {
    return 1;
  }

  static bool is_finite(const T& x) {
    return std::isfinite(x);
  }

  template<typename TReal>
  static auto euclidean_squared(const T& x) {
    return grex::convert_unsafe<TReal>(x) * grex::convert_unsafe<TReal>(x);
  }
  template<typename TReal>
  static auto max_norm(const T& x) {
    return compat::abs(grex::convert_unsafe<TReal>(x));
  }

  static T transpose(const T& value) {
    return value;
  }
  template<typename TSol>
  static TSol cholesky_lower(const T& value) {
    return compat::sqrt(TSol(value));
  }

  template<typename TReal>
  struct LuDecomposition {
    TReal u;

    constexpr TReal lower() const {
      return 1;
    }
    constexpr TReal upper() const {
      return u;
    }
  };
  template<typename TReal>
  static LuDecomposition<TReal> lu_decompose(const T& value) {
    return {TReal(value)};
  }
};
template<typename T>
inline T zero() {
  return CompatTrait<T>::zero();
}
template<typename T>
inline T signaling_nan() {
  return CompatTrait<T>::signaling_nan();
}
template<typename T>
inline T identity() {
  return CompatTrait<T>::identity();
}
template<typename T>
inline bool is_finite(const T& x) {
  return CompatTrait<T>::is_finite(x);
}
template<typename TReal, typename T>
inline auto euclidean_squared(const T& x) {
  return CompatTrait<T>::template euclidean_squared<TReal>(x);
}
template<typename TReal, typename T>
inline auto max_norm(const T& x) {
  return CompatTrait<T>::template max_norm<TReal>(x);
}
template<typename T>
inline auto transpose(T&& x) {
  return CompatTrait<std::decay_t<T>>::transpose(std::forward<T>(x));
}
template<typename TSol, typename T>
inline auto cholesky_lower(const T& x) {
  return CompatTrait<T>::template cholesky_lower<TSol>(x);
}
template<typename TReal, typename T>
inline auto lu_decompose(const T& x) {
  return CompatTrait<T>::template lu_decompose<TReal>(x);
}

template<typename T1, typename T2>
struct BiCompatTrait {
  template<typename TReal>
  static auto dot(const T1& x, const T2& y) {
    return grex::convert_unsafe<TReal>(x) * grex::convert_unsafe<TReal>(y);
  }
};
template<typename TReal, typename T1, typename T2>
inline auto dot(const T1& x, const T2& y) {
  return BiCompatTrait<T1, T2>::template dot<TReal>(x, y);
}

template<typename TWork, typename TLhs, typename TRhs>
struct SolveTrait {
  static constexpr TWork solve(const TLhs& lhs, const TRhs& rhs) {
    return TWork(rhs) / TWork(lhs);
  }
  static constexpr TWork solve_tri(const TLhs& lhs, const TRhs& rhs,
                                   AnyTriangularKindTag auto /*kind*/,
                                   AnyLhsHasUnitDiagonalTag auto /*lhs_has_unit_diagonal*/) {
    return TWork(rhs) / TWork(lhs);
  }
  static constexpr TWork solve_right(const TLhs& lhs, const TRhs& rhs) {
    return TWork(rhs) / TWork(lhs);
  }
  static constexpr TWork solve_right_tri(const TLhs& lhs, const TRhs& rhs,
                                         AnyTriangularKindTag auto /*kind*/,
                                         AnyLhsHasUnitDiagonalTag auto /*lhs_has_unit_diagonal*/) {
    return TWork(rhs) / TWork(lhs);
  }
};
template<typename TWork, typename TLhs, typename TRhs>
constexpr auto solve(const TLhs& lhs, const TRhs& rhs) {
  return SolveTrait<TWork, TLhs, TRhs>::solve(lhs, rhs);
}
template<typename TWork, typename TLhs, typename TRhs>
constexpr auto solve_tri(const TLhs& lhs, const TRhs& rhs, AnyTriangularKindTag auto kind,
                         AnyLhsHasUnitDiagonalTag auto lhs_has_unit_diagonal) {
  return SolveTrait<TWork, TLhs, TRhs>::solve_tri(lhs, rhs, kind, lhs_has_unit_diagonal);
}
template<typename TWork, typename TLhs, typename TRhs>
constexpr auto solve_right(const TLhs& lhs, const TRhs& rhs) {
  return SolveTrait<TWork, TLhs, TRhs>::solve_right(lhs, rhs);
}
template<typename TWork, typename TLhs, typename TRhs>
constexpr auto solve_right_tri(const TLhs& lhs, const TRhs& rhs, AnyTriangularKindTag auto kind,
                               AnyLhsHasUnitDiagonalTag auto lhs_has_unit_diagonal) {
  return SolveTrait<TWork, TLhs, TRhs>::solve_right_tri(lhs, rhs, kind, lhs_has_unit_diagonal);
}

template<typename TDst, typename TSrc>
struct CastTrait {
  static constexpr auto cast(TSrc value) {
    return grex::convert_unsafe<TDst>(value);
  }
};
template<typename TDst, typename TSrc>
[[nodiscard]] constexpr auto cast(TSrc&& src) {
  return CastTrait<TDst, TSrc>::cast(std::forward<TSrc>(src));
}

// GrexTaggedTrait

template<typename T>
struct GrexTaggedTrait {
  static T broadcast(T value, grex::OptValuedScalarTag<T> auto /*tag*/) {
    return value;
  }

  static void store(T* dst, const T& src, grex::OptValuedScalarTag<T> auto /*tag*/) {
    *dst = src;
  }

  static const T& load(const T* src, grex::OptValuedScalarTag<T> auto /*tag*/) {
    return *src;
  }
  static T& load(T* src, grex::OptValuedScalarTag<T> auto /*tag*/) {
    return *src;
  }

  static const T& load_extended(const T* src, grex::OptValuedScalarTag<T> auto /*tag*/) {
    return *src;
  }
  static T& load_extended(T* src, grex::OptValuedScalarTag<T> auto /*tag*/) {
    return *src;
  }
};

template<grex::Vectorizable T>
struct GrexTaggedTrait<T> {
  template<grex::AnyTag TTag>
  static auto broadcast(T value, TTag tag) {
    return grex::broadcast(value, tag);
  }

  template<grex::AnyTag TTag>
  static void store(T* dst, const grex::TagType<TTag, T> src, TTag tag) {
    grex::store(dst, src, tag);
  }

  static decltype(auto) load(const T* src, grex::AnyTag auto tag) {
    return grex::load(src, tag);
  }
  static decltype(auto) load(T* src, grex::AnyTag auto tag) {
    return grex::load(src, tag);
  }

  static decltype(auto) load_extended(const T* src, grex::AnyTag auto tag) {
    return grex::load_extended(src, tag);
  }
  static decltype(auto) load_extended(T* src, grex::AnyTag auto tag) {
    return grex::load_extended(src, tag);
  }
};

template<typename T>
inline decltype(auto) broadcast(T value, grex::AnyTag auto tag)
requires(requires { GrexTaggedTrait<T>::broadcast(value, tag); })
{
  return GrexTaggedTrait<T>::broadcast(value, tag);
}
template<typename T>
inline void store(T* dst, auto src, grex::AnyTag auto tag) {
  return GrexTaggedTrait<T>::store(dst, src, tag);
}
template<typename T>
inline decltype(auto) load(T* src, grex::AnyTag auto tag) {
  return GrexTaggedTrait<std::decay_t<T>>::load(src, tag);
}
template<typename T>
inline decltype(auto) load_extended(T* src, grex::AnyTag auto tag) {
  return GrexTaggedTrait<std::decay_t<T>>::load_extended(src, tag);
}
} // namespace lineal::compat

#endif // INCLUDE_LINEAL_BASE_COMPAT_OPERATION_HPP
