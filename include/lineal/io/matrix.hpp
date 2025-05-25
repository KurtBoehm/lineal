// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_IO_MATRIX_HPP
#define INCLUDE_LINEAL_IO_MATRIX_HPP

#include <functional>
#include <type_traits>
#include <utility>

#include "thesauros/format.hpp"
#include "thesauros/io.hpp"

#include "lineal/base.hpp"

namespace lineal {
template<SharedMatrix TMatrix, typename TRowTrans, typename TColTrans, bool tSkipZero, bool tFormat,
         bool tIsOrdered, bool tExtended>
struct MatrixPrinter {
  static constexpr auto fmt_tag = thes::FormattingTag<tFormat>{};

  TMatrix matrix;
  [[no_unique_address]] TRowTrans row_trans;
  [[no_unique_address]] TColTrans col_trans;
};

template<SharedMatrix TMatrix, typename TRowOff, typename TColOff, bool tSkipZero, bool tFormat,
         bool tIsOrdered, bool tOwn>
[[nodiscard]] inline auto
matrix_print_base(TMatrix&& matrix, TRowOff row_off, TColOff col_off,
                  ZeroSkippingTag<tSkipZero> /*skip_zero*/, thes::FormattingTag<tFormat> /*format*/,
                  OrderingTag<tIsOrdered> /*ordered*/, LocalPartTag<tOwn> /*part*/) {
  return MatrixPrinter<TMatrix, TRowOff, TColOff, tSkipZero, tFormat, tIsOrdered, !tOwn>{
    std::forward<TMatrix>(matrix), row_off, col_off};
}

template<SharedMatrix TMatrix>
[[nodiscard]] inline auto matrix_print(TMatrix&& matrix, AnyZeroSkippingTag auto skip_zero,
                                       thes::AnyFormatTag auto format,
                                       AnyOrderingTag auto ordered) {
  constexpr auto noop = std::identity{};
  return matrix_print_base(std::forward<TMatrix>(matrix), noop, noop, skip_zero, format, ordered,
                           own_tag);
}
} // namespace lineal

template<lineal::SharedMatrix TMatrix, typename TRowTrans, typename TColTrans, bool tSkipZero,
         bool tFormat, bool tIsOrdered, bool tExtended>
struct fmt::formatter<
  lineal::MatrixPrinter<TMatrix, TRowTrans, TColTrans, tSkipZero, tFormat, tIsOrdered, tExtended>>
    : public fmt::nested_formatter<typename std::decay_t<TMatrix>::Value> {
  using Self =
    lineal::MatrixPrinter<TMatrix, TRowTrans, TColTrans, tSkipZero, tFormat, tIsOrdered, tExtended>;
  static constexpr auto fmt_tag = thes::FormattingTag<tFormat>{};

  auto format(const Self& printer, fmt::format_context& ctx) const {
    return this->write_padded(ctx, [&](auto it) {
      decltype(auto) range = [&]() -> decltype(auto) {
        if constexpr (tExtended) {
          return printer.matrix.ext_range();
        } else {
          return printer.matrix;
        }
      }();

      for (const auto& row : range) {
        const auto i = printer.row_trans(row.index());
        it = fmt::format_to(
          it, "{}: ", thes::tstyled(fmt_tag, thes::numeric_string(i).value(), thes::fg_red));

        thes::Delimiter delim{", "};
        auto op = [&](const auto col, const auto v, auto is_diag) {
          if constexpr (tSkipZero) {
            if (v == 0) {
              return;
            }
          }

          const auto j = printer.col_trans(col);
          it =
            fmt::format_to(it, "{}{} → {}", delim,
                           thes::tstyled(fmt_tag, thes::numeric_string(j).value(),
                                         is_diag ? thes::fg_yellow : thes::fg_green),
                           thes::tstyled(fmt_tag, v, is_diag ? thes::fg_magenta : thes::fg_blue));
        };

        row.iterate([&](const auto col, const auto v) { op(col, v, false); },
                    [&](const auto col, const auto v) { op(col, v, true); },
                    [&](const auto col, const auto v) { op(col, v, false); }, lineal::valued_tag,
                    lineal::OrderingTag<tIsOrdered>{});
        *it++ = '\n';
      }
      return it;
    });
  }
};

#endif // INCLUDE_LINEAL_IO_MATRIX_HPP
