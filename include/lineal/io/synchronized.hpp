// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_IO_SYNCHRONIZED_HPP
#define INCLUDE_LINEAL_IO_SYNCHRONIZED_HPP

#include <cstdio>

#include "thesauros/format.hpp"

namespace lineal {
template<typename... TArgs>
inline auto sync_print(fmt::format_string<TArgs...> format_string, TArgs&&... args) {
  fmt::print(format_string, std::forward<TArgs>(args)...);
}
template<typename TS, typename... TArgs>
inline auto sync_print(const fmt::text_style& ts, const TS& s, TArgs&&... args) {
  fmt::print(ts, s, std::forward<TArgs>(args)...);
}
template<typename... TArgs>
inline auto sync_print(FILE* file, fmt::format_string<TArgs...> format_string, TArgs&&... args) {
  fmt::print(file, format_string, std::forward<TArgs>(args)...);
}
template<typename TS, typename... TArgs>
inline auto sync_print(FILE* file, const fmt::text_style& ts, const TS& s, TArgs&&... args) {
  fmt::print(file, ts, s, std::forward<TArgs>(args)...);
}
} // namespace lineal

#endif // INCLUDE_LINEAL_IO_SYNCHRONIZED_HPP
