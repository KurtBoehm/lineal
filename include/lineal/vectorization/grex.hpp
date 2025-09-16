// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_VECTORIZATION_GREX_HPP
#define INCLUDE_LINEAL_VECTORIZATION_GREX_HPP

#include <type_traits>

#include "grex/grex.hpp" // IWYU pragma: export
#include "thesauros/utility/integral-value.hpp"

namespace grex {
template<typename TTag>
struct GeometryRespectingTag : public TTag {
  static constexpr bool is_geometry_respecting = true;

  using TTag::TTag;
  explicit constexpr GeometryRespectingTag(TTag tag) : TTag(tag) {}
};

template<typename TTag>
struct TagTraits<GeometryRespectingTag<TTag>> : public TagTraits<TTag> {};

// tag traits
template<typename TTag>
struct IsGeometryRespectingTrait : public std::false_type {};
template<typename TTag>
requires(requires { TTag::is_geometry_respecting; })
struct IsGeometryRespectingTrait<TTag> : public std::bool_constant<TTag::is_geometry_respecting> {};
template<typename TTag>
inline constexpr bool is_geometry_respecting = IsGeometryRespectingTrait<TTag>::value;
} // namespace grex

// Allow grex::ValueTag to be converted to an integer using Thesauros’ utilities, too
template<typename T, T tValue>
struct thes::IntegralValueTrait<grex::ValueTag<T, tValue>> {
  using Type = IntegralValueTrait<T>::Type;
  static constexpr Type value(const grex::ValueTag<T, tValue> tag) GREX_ALWAYS_INLINE {
    return IntegralValueTrait<T>::value(tag.value);
  }
};

#endif // INCLUDE_LINEAL_VECTORIZATION_GREX_HPP
