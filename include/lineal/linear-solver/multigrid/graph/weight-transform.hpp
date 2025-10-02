// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_GRAPH_WEIGHT_TRANSFORM_HPP
#define INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_GRAPH_WEIGHT_TRANSFORM_HPP

#include "thesauros/macropolis.hpp"

#include "lineal/tensor/fixed/matrix.hpp"
#include "lineal/vectorization.hpp"

namespace lineal::amg {
struct NoOpTransform {
  template<typename T>
  using Value = T;

  THES_ALWAYS_INLINE static constexpr auto diagonal(const auto& value) {
    return value;
  }
  THES_ALWAYS_INLINE static constexpr auto offdiagonal(const auto& value) {
    return value;
  }
};

struct NegativeOffdiagonalTransform {
  template<typename T>
  using Value = T;

  THES_ALWAYS_INLINE static constexpr auto diagonal(const auto& value) {
    return value;
  }
  THES_ALWAYS_INLINE static constexpr auto offdiagonal(const auto& value) {
    return -value;
  }
};

struct NegativeDiagonalTransform {
  template<typename T>
  using Value = T;

  THES_ALWAYS_INLINE static constexpr auto diagonal(const auto& value) {
    return -value;
  }
  THES_ALWAYS_INLINE static constexpr auto offdiagonal(const auto& value) {
    return value;
  }
};

struct AbsTransform {
  template<typename T>
  using Value = T;

  THES_ALWAYS_INLINE static constexpr auto diagonal(const auto& value) {
    return grex::abs(value);
  }
  THES_ALWAYS_INLINE static constexpr auto offdiagonal(const auto& value) {
    return grex::abs(value);
  }
};

template<typename TReal>
struct FrobeniusTransform {
  template<typename T>
  using Value = TReal;

  THES_ALWAYS_INLINE static constexpr auto diagonal(const auto& value) {
    return fix::frobenius_norm<TReal>(value);
  }
  THES_ALWAYS_INLINE static constexpr auto offdiagonal(const auto& value) {
    return fix::frobenius_norm<TReal>(value);
  }
};
} // namespace lineal::amg

#endif // INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_GRAPH_WEIGHT_TRANSFORM_HPP
