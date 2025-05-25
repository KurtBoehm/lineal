// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_IO_GREX_HPP
#define INCLUDE_LINEAL_IO_GREX_HPP

#include <array>
#include <cstddef>

#include "thesauros/format.hpp"
#include "thesauros/ranges.hpp"

#include "lineal/vectorization/grex.hpp"

template<grex::AnyVector TVec>
struct fmt::formatter<TVec>
    : public fmt::nested_formatter<std::array<typename TVec::Value, TVec::size>> {
  using Value = TVec::Value;
  static constexpr std::size_t size = TVec::size;

  auto format(TVec vec, fmt::format_context& ctx) const {
    std::array<Value, size> data;
    vec.store(data.data());
    return this->write_padded(ctx, [&](auto it) { return fmt::format_to(it, "{}", data); });
  }
};

template<grex::AnyMask TMask>
struct fmt::formatter<TMask> : public fmt::nested_formatter<std::array<bool, TMask::size>> {
  static constexpr std::size_t size = TMask::size;

  auto format(TMask mask, fmt::format_context& ctx) const {
    std::array<bool, size> data;
    for (const std::size_t i : thes::range(size)) {
      data[i] = mask[i];
    }
    return this->write_padded(ctx, [&](auto it) { return fmt::format_to(it, "{}", data); });
  }
};

#endif // INCLUDE_LINEAL_IO_GREX_HPP
