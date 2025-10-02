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
#include "lineal/tensor/fixed/matrix.hpp"

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
    return thes::star::index_transform<dot_num>(
             [&](auto j) { return Value(lhs_[i, j]) * Value(rhs_[j]); }) |
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
  return MatrixVectorProductView<thes::Union<LhsValue, RhsValue>, TLhs, TRhs>{
    std::forward<TLhs>(lhs), std::forward<TRhs>(rhs)};
}

template<typename TValue, AnyMatrix TLhs, AnyMatrix TRhs>
requires(std::decay_t<TLhs>::dimensions.column_num == std::decay_t<TRhs>::dimensions.row_num)
struct MatrixMatrixProductView : public MatrixBase {
  using Value = TValue;
  using Lhs = std::decay_t<TLhs>;
  using Rhs = std::decay_t<TRhs>;

  static constexpr std::size_t row_num = Lhs::dimensions.row_num;
  static constexpr std::size_t colum_num = Rhs::dimensions.column_num;
  static constexpr MatrixDimensions dimensions{.row_num = row_num, .column_num = colum_num};

  explicit constexpr MatrixMatrixProductView(TLhs&& lhs, TRhs&& rhs)
      : lhs_(std::forward<TLhs>(lhs)), rhs_(std::forward<TRhs>(rhs)) {}

  template<std::size_t tRow, std::size_t tCol>
  requires(tRow < row_num && tCol < colum_num)
  constexpr Value operator[](thes::IndexTag<tRow> i, thes::IndexTag<tCol> j) const {
    Value dst{0};
    thes::star::iota<0, Lhs::dimensions.column_num> |
      thes::star::for_each([&](auto k) { dst += lhs_[i, k] * rhs_[k, j]; });
    return dst;
  }
  constexpr Value operator[](std::size_t i, std::size_t j) const {
    return thes::star::static_apply<Lhs::dimensions.column_num>(
      [&]<std::size_t... tK>() { return (Value{0} + ... + (lhs_[i, tK] * rhs_[tK, j])); });
  }

private:
  TLhs lhs_;
  TRhs rhs_;
};

template<AnyMatrix TLhs, AnyMatrix TRhs>
requires(std::decay_t<TLhs>::dimensions.column_num == std::decay_t<TRhs>::dimensions.row_num)
constexpr auto operator*(TLhs&& lhs, TRhs&& rhs) {
  using LhsValue = std::decay_t<TLhs>::Value;
  using RhsValue = std::decay_t<TRhs>::Value;
  return MatrixMatrixProductView<thes::Union<LhsValue, RhsValue>, TLhs, TRhs>{
    std::forward<TLhs>(lhs), std::forward<TRhs>(rhs)};
}
} // namespace lineal::fix

#endif // INCLUDE_LINEAL_TENSOR_FIXED_EXPRESSIONS_HPP
