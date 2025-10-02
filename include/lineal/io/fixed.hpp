// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_IO_FIXED_HPP
#define INCLUDE_LINEAL_IO_FIXED_HPP

#include <type_traits>

#include "thesauros/format.hpp"
#include "thesauros/io/delimiter.hpp"

#include "lineal/tensor/fixed/concepts.hpp"

template<lineal::fix::AnyVector TVector>
struct fmt::range_format_kind<TVector, char> {
  static constexpr auto value = fmt::range_format::disabled;
};

template<lineal::fix::AnyVector TVector>
struct fmt::formatter<TVector> : public fmt::formatter<typename std::decay_t<TVector>::Value> {
  using ValueFmt = fmt::formatter<typename std::decay_t<TVector>::Value>;

  auto format(const TVector& vec, fmt::format_context& ctx) const {
    thes::Delimiter delim{", "};
    auto it = ctx.out();
    *it++ = '[';
    thes::star::iota<0, vec.size> | thes::star::for_each([&](auto i) {
      it = fmt::format_to(it, "{}", delim);
      ctx.advance_to(it);
      it = ValueFmt::format(vec[i], ctx);
    });
    *it++ = ']';
    return it;
  }
};

template<lineal::fix::AnyMatrix TMatrix>
struct fmt::formatter<TMatrix> : public fmt::formatter<typename std::decay_t<TMatrix>::Value> {
  using ValueFmt = fmt::formatter<typename std::decay_t<TMatrix>::Value>;

  auto format(const TMatrix& mat, fmt::format_context& ctx) const {
    thes::Delimiter rd{", "};
    auto it = ctx.out();
    *it++ = '[';
    thes::star::iota<0, mat.dimensions.row_num> | thes::star::for_each([&](auto i) {
      it = fmt::format_to(it, "{}", rd);
      *it++ = '[';
      thes::Delimiter cd{","};
      thes::star::iota<0, mat.dimensions.column_num> | thes::star::for_each([&](auto j) {
        it = fmt::format_to(it, "{}", cd);
        ctx.advance_to(it);
        it = ValueFmt::format(mat[i, j], ctx);
      });
      *it++ = ']';
    });
    *it++ = ']';
    return it;
  }
};

#endif // INCLUDE_LINEAL_IO_FIXED_HPP
