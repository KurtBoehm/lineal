// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_GRAPH_WEIGHT_TRANSFORM_HPP
#define INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_GRAPH_WEIGHT_TRANSFORM_HPP

#include "thesauros/macropolis.hpp"

#include "lineal/vectorization.hpp"

namespace lineal::amg {
struct NegativeOffdiagonalTransform {
  THES_ALWAYS_INLINE static constexpr auto diagonal(const auto& value, grex::AnyTag auto /*tag*/) {
    return value;
  }
  THES_ALWAYS_INLINE static constexpr auto offdiagonal(const auto& value,
                                                       grex::AnyTag auto /*tag*/) {
    return -value;
  }
};

struct NegativeDiagonalTransform {
  THES_ALWAYS_INLINE static constexpr auto diagonal(const auto& value, grex::AnyTag auto /*tag*/) {
    return -value;
  }
  THES_ALWAYS_INLINE static constexpr auto offdiagonal(const auto& value,
                                                       grex::AnyTag auto /*tag*/) {
    return value;
  }
};

struct AbsTransform {
  THES_ALWAYS_INLINE static constexpr auto diagonal(const auto& value, grex::AnyTag auto tag) {
    return grex::abs(value, tag);
  }
  THES_ALWAYS_INLINE static constexpr auto offdiagonal(const auto& value, grex::AnyTag auto tag) {
    return grex::abs(value, tag);
  }
};
} // namespace lineal::amg

#endif // INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_GRAPH_WEIGHT_TRANSFORM_HPP
