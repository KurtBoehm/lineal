// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_TENSOR_DYNAMIC_STENCIL_DEF_HPP
#define INCLUDE_LINEAL_TENSOR_DYNAMIC_STENCIL_DEF_HPP

#include "thesauros/math.hpp"
#include "thesauros/ranges.hpp"
#include "thesauros/types.hpp"

namespace lineal {
enum struct AxisSide : bool { START, END };

template<thes::AnyIotaRange TRange>
constexpr auto material_range(const auto& sys_info, const TRange& range) {
  if (sys_info.total_size() == range.size()) {
    return range;
  }
  const auto offset = sys_info.after_size(thes::index_tag<0>);

  using Size = TRange::Value;
  const auto begin = thes::sub_min<Size>(range.begin_value(), offset, 0);
  const auto end = thes::add_max<Size>(range.end_value(), offset, sys_info.total_size());

  return TRange{begin, end};
}
} // namespace lineal

#endif // INCLUDE_LINEAL_TENSOR_DYNAMIC_STENCIL_DEF_HPP
