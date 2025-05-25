// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_TENSOR_FIXED_CONCEPTS_HPP
#define INCLUDE_LINEAL_TENSOR_FIXED_CONCEPTS_HPP

#include <concepts>
#include <type_traits>

namespace lineal::fix {
struct VectorBase {
  constexpr bool operator==(const VectorBase& other) const = default;
};
template<typename T>
struct IsVectorTrait : public std::is_base_of<VectorBase, T> {};
template<typename T>
concept AnyVector = IsVectorTrait<std::decay_t<T>>::value;

struct MatrixBase {};
template<typename T>
struct IsMatrixTrait : public std::is_base_of<MatrixBase, T> {};
template<typename T>
concept AnyMatrix = IsMatrixTrait<std::decay_t<T>>::value;

template<typename T>
concept AnyLowerTriangularMatrix = AnyMatrix<T> && requires {
  typename std::decay_t<T>::Kind;
  { std::decay_t<T>::Kind::is_lower_triangular } -> std::convertible_to<bool>;
  requires std::decay_t<T>::Kind::is_lower_triangular;
};

template<typename T>
concept AnyUpperTriangularMatrix = AnyMatrix<T> && requires {
  typename std::decay_t<T>::Kind;
  { std::decay_t<T>::Kind::is_upper_triangular } -> std::convertible_to<bool>;
  requires std::decay_t<T>::Kind::is_upper_triangular;
};

template<typename T>
concept AnySymmetricMatrix = AnyMatrix<T> && requires {
  typename std::decay_t<T>::Kind;
  { std::decay_t<T>::Kind::is_symmetric } -> std::convertible_to<bool>;
  requires std::decay_t<T>::Kind::is_symmetric;
};
} // namespace lineal::fix

#endif // INCLUDE_LINEAL_TENSOR_FIXED_CONCEPTS_HPP
