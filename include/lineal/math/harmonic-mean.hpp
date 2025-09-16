// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_MATH_HARMONIC_MEAN_HPP
#define INCLUDE_LINEAL_MATH_HARMONIC_MEAN_HPP

#include "lineal/vectorization.hpp"

namespace lineal {
template<grex::FloatVectorizable T>
inline T harmonic_mean(const T a, const T b) {
  constexpr auto tag = grex::full_tag<grex::native_sizes<T>.front()>;
  const T p = a * b;
  return grex::mask_divide(grex::expand_zero(p, tag) != grex::zeros<T>(tag),
                           grex::expand_any(p + p, tag), grex::expand_any(a + b, tag))[0];
}

template<grex::AnyVector TVec>
inline TVec harmonic_mean(const TVec a, const TVec b) {
  const TVec p = a * b;
  return grex::mask_divide(p != TVec{}, p + p, a + b);
}
} // namespace lineal

#endif // INCLUDE_LINEAL_MATH_HARMONIC_MEAN_HPP
