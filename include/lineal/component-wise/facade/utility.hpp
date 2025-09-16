// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_COMPONENT_WISE_FACADE_UTILITY_HPP
#define INCLUDE_LINEAL_COMPONENT_WISE_FACADE_UTILITY_HPP

#include <cassert>

#include "thesauros/static-ranges.hpp"
#include "thesauros/types.hpp"

namespace lineal::impl {
constexpr auto axis_range_children(const auto& children, thes::AnyIndexTag auto idx) {
  return children | thes::star::filter<[](auto /*idx*/, auto type) {
           return requires(const decltype(type)::Type& vec) { vec.axis_range(idx); };
         }>;
}

constexpr auto axis_range(const auto& children, thes::AnyIndexTag auto idx)
requires(!thes::star::is_empty<decltype(axis_range_children(children, idx))>)
{
  auto axis_ranges =
    axis_range_children(children, idx) |
    thes::star::transform([idx](const auto& c) -> decltype(auto) { return c.axis_range(idx); });
  assert(axis_ranges | thes::star::has_unique_value);
  return thes::star::get_at<0>(axis_ranges);
}

constexpr auto geometry_children(const auto& children) {
  return children | thes::star::filter<[](auto /*idx*/, auto type) {
           return requires(const decltype(type)::Type& vec) { vec.geometry(); };
         }>;
}
constexpr decltype(auto) geometry(const auto& children)
requires(!thes::star::is_empty<decltype(geometry_children(children))>)
{
  auto geometries =
    geometry_children(children) |
    thes::star::transform([](const auto& c) -> decltype(auto) { return c.geometry(); });
  assert(geometries | thes::star::has_unique_value);
  return thes::star::get_at<0>(geometries);
}
} // namespace lineal::impl

#endif // INCLUDE_LINEAL_COMPONENT_WISE_FACADE_UTILITY_HPP
