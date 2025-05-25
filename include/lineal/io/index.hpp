// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_IO_INDEX_HPP
#define INCLUDE_LINEAL_IO_INDEX_HPP

#include <concepts>
#include <type_traits>

#include "thesauros/format.hpp"

#include "lineal/parallel.hpp"

template<std::unsigned_integral TSize>
struct fmt::formatter<lineal::IndexRangeMap<TSize>> : public fmt::nested_formatter<TSize> {
  auto format(lineal::IndexRangeMap<TSize> map, fmt::format_context& ctx) const {
    return this->write_padded(ctx, [&](auto it) {
      return fmt::format_to(it, "{}:{}; {}", map.begin(), map.end(),
                            std::bit_cast<std::make_signed_t<TSize>>(map.offset()));
    });
  }
};

#endif // INCLUDE_LINEAL_IO_INDEX_HPP
