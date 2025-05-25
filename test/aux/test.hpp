// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef TEST_AUX_TEST_HPP
#define TEST_AUX_TEST_HPP

#include <concepts>
#include <functional>

#include "thesauros/format.hpp"
#include "thesauros/static-ranges.hpp"
#include "thesauros/test.hpp"
#include "thesauros/types.hpp"

#include "lineal/io.hpp"
#include "lineal/parallel.hpp"

namespace lineal::test {
template<typename TVec>
inline decltype(auto) trans_vector(TVec&& vec, AnyIndexTag auto tag) {
  return std::forward<TVec>(vec);
}

struct ComparisonPrinter {
  template<typename... TArgs>
  void operator()(::fmt::format_string<TArgs...> fmt, TArgs&&... args) {
    sync_print(fmt, std::forward<TArgs>(args)...);
  }

  template<typename... TArgs>
  void operator()(const ::fmt::text_style& ts, ::fmt::format_string<TArgs...> fmt,
                  TArgs&&... args) {
    sync_print(ts, fmt, std::forward<TArgs>(args)...);
  }
};

template<typename TVec1, typename TVec2, thes::AnyBoolTag TVerbose = thes::BoolTag<false>>
inline bool vector_eq(TVec1&& vec1, TVec2&& vec2, TVerbose verbose = {}) {
  if constexpr (verbose) {
    return thes::test::range_eq(std::forward<TVec1>(vec1), std::forward<TVec2>(vec2), {},
                                ComparisonPrinter{});
  } else {
    return thes::test::range_eq(std::forward<TVec1>(vec1), std::forward<TVec2>(vec2), {});
  }
}
template<typename TVec1, typename TVec2, thes::AnyBoolTag TVerbose = thes::BoolTag<false>>
inline bool vector_eq(TVec1&& vec1, TVec2&& vec2, AnyIndexTag auto tag, TVerbose verbose = {}) {
  return vector_eq(trans_vector(std::forward<TVec1>(vec1), tag),
                   trans_vector(std::forward<TVec2>(vec2), tag), verbose);
}

template<typename TVec1, typename TVec2, thes::AnyBoolTag TVerbose = thes::BoolTag<false>>
inline bool vector_eq(TVec1&& vec1, TVec2&& vec2, std::floating_point auto diff,
                      TVerbose verbose = {}) {
  auto op = [diff](auto v1, auto v2) { return std::abs(v1 - v2) <= diff; };
  if constexpr (verbose) {
    return thes::test::range_eq(std::forward<TVec1>(vec1), std::forward<TVec2>(vec2), op,
                                ComparisonPrinter{});
  } else {
    return thes::test::range_eq(std::forward<TVec1>(vec1), std::forward<TVec2>(vec2), op);
  }
}
template<typename TVec1, typename TVec2, thes::AnyBoolTag TVerbose = thes::BoolTag<false>>
inline bool vector_eq(TVec1&& vec1, TVec2&& vec2, AnyIndexTag auto tag,
                      std::floating_point auto diff, TVerbose verbose = {}) {
  return vector_eq(trans_vector(std::forward<TVec1>(vec1), tag),
                   trans_vector(std::forward<TVec2>(vec2), tag), diff, verbose);
}

inline bool star_eq(const auto& r1, const auto& r2) {
  return thes::star::zip(r1, r2) | thes::star::transform([](auto p) {
           return thes::star::get_at<0>(p) == thes::star::get_at<1>(p);
         }) |
         thes::star::left_reduce(std::logical_and<>{});
}
} // namespace lineal::test

#endif // TEST_AUX_TEST_HPP
