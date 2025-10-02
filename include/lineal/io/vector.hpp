// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_IO_VECTOR_HPP
#define INCLUDE_LINEAL_IO_VECTOR_HPP

#include <cstddef>
#include <type_traits>
#include <utility>

#include "thesauros/format.hpp"
#include "thesauros/io.hpp"

#include "lineal/base.hpp"

namespace lineal {
template<AnyVector TVec>
struct VectorPrinter {
  explicit VectorPrinter(TVec&& vec) : vec_(std::forward<TVec>(vec)) {}

  const TVec& vec() const {
    return vec_;
  }

private:
  TVec vec_;
};
template<AnyVector TVec>
VectorPrinter(TVec&&) -> VectorPrinter<TVec>;

template<AnyVector TVec>
inline auto vector_print(TVec&& vec) {
  return VectorPrinter<TVec>(std::forward<TVec>(vec));
}
} // namespace lineal

template<lineal::AnyVector TVec>
struct fmt::formatter<lineal::VectorPrinter<TVec>>
    : public fmt::nested_formatter<typename std::decay_t<TVec>::Value> {
  auto format(const lineal::VectorPrinter<TVec>& printer, fmt::format_context& ctx) const {
    return this->write_padded(ctx, [&](auto it) {
      auto range_print = [&](const auto& range) {
        it = fmt::format_to(it, "[");
        for (thes::Delimiter d{", "}; const auto v : range) {
          const auto fg = [&] {
            if constexpr (requires { std::size_t(v); }) {
              return thes::rainbow_fg(std::size_t(v));
            } else {
              return thes::fg_blue;
            }
          }();
          it = fmt::format_to(it, "{}{}", d, fmt::styled(this->nested(v), fg));
        }
        return fmt::format_to(it, "]");
      };

      return range_print(printer.vec());
    });
  }
};

#endif // INCLUDE_LINEAL_IO_VECTOR_HPP
