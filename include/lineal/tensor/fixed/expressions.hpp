// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_TENSOR_FIXED_EXPRESSIONS_HPP
#define INCLUDE_LINEAL_TENSOR_FIXED_EXPRESSIONS_HPP

#include <cassert>
#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>

#include "thesauros/types.hpp"

#include "lineal/tensor/fixed/concepts.hpp"

namespace lineal::fix {
template<typename TValue, AnyMatrix TLhs, AnyVector TRhs>
requires(std::decay_t<TLhs>::dimensions.column_num == std::decay_t<TRhs>::size)
struct MatrixVectorProductView : public VectorBase {
  using Lhs = std::decay_t<TLhs>;
  using Value = TValue;
  static constexpr std::size_t dot_num = Lhs::dimensions.column_num;
  static constexpr std::size_t size = Lhs::dimensions.row_num;

  explicit constexpr MatrixVectorProductView(TLhs&& lhs, TRhs&& rhs)
      : lhs_(std::forward<TLhs>(lhs)), rhs_(std::forward<TRhs>(rhs)) {}

  template<std::size_t tIdx>
  requires(tIdx < size)
  constexpr Value operator[](thes::IndexTag<tIdx> i) const {
    return thes::star::index_transform<dot_num>([&](auto j) { return lhs_(i, j) * rhs_[j]; }) |
           thes::star::left_reduce(std::plus{}, TValue{});
  }

private:
  TLhs lhs_;
  TRhs rhs_;
};

template<AnyMatrix TLhs, AnyVector TRhs>
constexpr auto operator*(TLhs&& lhs, TRhs&& rhs) {
  using LhsValue = std::decay_t<TLhs>::Value;
  using RhsValue = std::decay_t<TRhs>::Value;
  return MatrixVectorProductView<std::common_type_t<LhsValue, RhsValue>, TLhs, TRhs>{
    std::forward<TLhs>(lhs), std::forward<TRhs>(rhs)};
}
} // namespace lineal::fix

#endif // INCLUDE_LINEAL_TENSOR_FIXED_EXPRESSIONS_HPP
