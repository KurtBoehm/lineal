// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_TENSOR_FIXED_TRAITS_HPP
#define INCLUDE_LINEAL_TENSOR_FIXED_TRAITS_HPP

#include <cstddef>

#include "lineal/base/type-trait/tensor.hpp"
#include "lineal/tensor/fixed/matrix.hpp"
#include "lineal/tensor/fixed/vector.hpp"

namespace lineal {
template<typename T, std::size_t tRows, std::size_t tCols>
struct ScalarTypeTrait<fix::DenseMatrix<T, tRows, tCols>> {
  using Type = T;
};
template<typename T, std::size_t tSize>
struct ScalarTypeTrait<fix::DenseVector<T, tSize>> {
  using Type = T;
};

template<typename TTarget, std::size_t tRows, std::size_t tCols, typename TReplaceWith>
struct WithScalarTypeTrait<fix::DenseMatrix<TTarget, tRows, tCols>, TReplaceWith> {
  using Type = fix::DenseMatrix<TReplaceWith, tRows, tCols>;
};
template<typename TTarget, std::size_t tSize, typename TReplaceWith>
struct WithScalarTypeTrait<fix::DenseVector<TTarget, tSize>, TReplaceWith> {
  using Type = fix::DenseVector<TReplaceWith, tSize>;
};
} // namespace lineal

#endif // INCLUDE_LINEAL_TENSOR_FIXED_TRAITS_HPP
