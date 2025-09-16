// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_TENSOR_DYNAMIC_STENCIL_VALUATOR_BASE_HPP
#define INCLUDE_LINEAL_TENSOR_DYNAMIC_STENCIL_VALUATOR_BASE_HPP

#include <cstddef>

#include "lineal/vectorization.hpp"

namespace lineal {
template<typename TVal>
struct ValuatorPtrIter {
  using Ptr = const TVal*;

  explicit constexpr ValuatorPtrIter(Ptr ptr) : ptr_(ptr) {}

  TVal operator*() const {
    return *ptr_;
  }
  auto load_ext(grex::AnyTag auto tag) const {
    return grex::load_extended(ptr_, tag);
  }
  auto load(grex::AnyTag auto tag) const {
    return grex::load(ptr_, tag);
  }

  ValuatorPtrIter operator+(std::ptrdiff_t off) const {
    return ValuatorPtrIter{ptr_ + off};
  }
  ValuatorPtrIter operator-(std::ptrdiff_t off) const {
    return ValuatorPtrIter{ptr_ - off};
  }

private:
  Ptr ptr_;
};
} // namespace lineal

#endif // INCLUDE_LINEAL_TENSOR_DYNAMIC_STENCIL_VALUATOR_BASE_HPP
