// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_BASE_ENUM_HPP
#define INCLUDE_LINEAL_BASE_ENUM_HPP

#include "thesauros/types/value-tag.hpp"

namespace lineal {
enum struct TriangularKind : bool { lower, upper };
constexpr TriangularKind operator!(TriangularKind tk) {
  switch (tk) {
    case TriangularKind::lower: return TriangularKind::upper;
    case TriangularKind::upper: return TriangularKind::lower;
  }
}
inline constexpr TriangularKind tri_lower = TriangularKind::lower;
inline constexpr thes::AutoTag<TriangularKind::lower> tri_lower_tag{};
inline constexpr TriangularKind tri_upper = TriangularKind::upper;
inline constexpr thes::AutoTag<TriangularKind::upper> tri_upper_tag{};
template<typename TValueTag>
concept AnyTriangularKindTag = thes::TypedValueTag<TValueTag, TriangularKind>;

enum struct LhsHasUnitDiagonal : bool {};
template<bool tValue>
inline constexpr LhsHasUnitDiagonal lhs_has_unit_diagonal = LhsHasUnitDiagonal{tValue};
template<bool tValue>
inline constexpr thes::AutoTag<LhsHasUnitDiagonal{tValue}> lhs_has_unit_diagonal_tag{};
template<typename TValueTag>
concept AnyLhsHasUnitDiagonalTag = thes::TypedValueTag<TValueTag, LhsHasUnitDiagonal>;
} // namespace lineal

#endif // INCLUDE_LINEAL_BASE_ENUM_HPP
