// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_TENSOR_DYNAMIC_MIX_UTILITY_HPP
#define INCLUDE_LINEAL_TENSOR_DYNAMIC_MIX_UTILITY_HPP

#include <cassert>

#include "lineal/base.hpp"

namespace lineal {
constexpr auto tensor_size(const SharedVector auto& src) {
  return src.size();
}
constexpr auto tensor_size(const SharedMatrix auto& src, MatrixRowTag /*tag*/) {
  return src.row_num();
}
constexpr auto tensor_size(const SharedMatrix auto& src, MatrixColumnTag /*tag*/) {
  return src.column_num();
}
constexpr auto tensor_size(const SharedMatrix auto& src) {
  assert(src.row_num() == src.column_num());
  return src.row_num();
}

template<SharedVector TDst>
constexpr TDst create_undef_like(const SharedTensor auto& src, auto... opt_tag) {
  return TDst(tensor_size(src, opt_tag...));
}

template<SharedVector TDst>
constexpr TDst create_numa_undef_like(const SharedTensor auto& src, const Env auto& env,
                                      auto... opt_tag) {
  return TDst(tensor_size(src, opt_tag...), env);
}

template<SharedVector TDst>
constexpr TDst create_constant_like(const SharedTensor auto& src, typename TDst::Value value,
                                    const Env auto& env, auto... opt_tag) {
  return TDst(tensor_size(src, opt_tag...), value, env);
}
} // namespace lineal

#endif // INCLUDE_LINEAL_TENSOR_DYNAMIC_MIX_UTILITY_HPP
