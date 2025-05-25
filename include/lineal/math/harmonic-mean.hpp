// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_MATH_HARMONIC_MEAN_HPP
#define INCLUDE_LINEAL_MATH_HARMONIC_MEAN_HPP

#include <concepts>
#include <cstddef>

#include "thesauros/math.hpp"
#include "thesauros/static-ranges.hpp"

#include "lineal/vectorization.hpp"

namespace lineal {
template<std::floating_point T>
inline T harmonic_mean(const T a, const T b, grex::Scalar /*tag*/) {
  constexpr std::size_t vec_size = grex::native_sizes<T> | thes::star::minimum;
  using Vector = grex::Vector<T, vec_size>;
  const T p = a * b;
  auto to_vec = [](T v) { return Vector(thes::fast::sse::to_sse(v)); };
  return grex::select_div(to_vec(p) != Vector{T{0}}, to_vec(p + p), to_vec(a + b),
                          grex::VectorSize<vec_size>{})[0];
}

template<grex::AnyVector TVec>
inline TVec harmonic_mean(const TVec a, const TVec b, grex::VectorTag auto tag) {
  using Real = TVec::Value;
  const TVec zero{Real{0}};
  const TVec p = a * b;
  const TVec s = a + b;
  return grex::select_div(p != zero, p + p, s, tag);
}
} // namespace lineal

#endif // INCLUDE_LINEAL_MATH_HARMONIC_MEAN_HPP
