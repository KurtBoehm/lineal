// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_BASE_TYPE_TRAIT_TENSOR_HPP
#define INCLUDE_LINEAL_BASE_TYPE_TRAIT_TENSOR_HPP

#include <cstddef>
#include <type_traits>

#include "grex/grex.hpp"
#include "thesauros/types.hpp"

namespace lineal {
template<typename T>
struct ScalarTypeTrait {
  using Type = T;
};
template<typename T>
using ScalarType = ScalarTypeTrait<T>::Type;

template<typename TTarget, typename TReplaceWith>
struct WithScalarTypeTrait {
  using Type = TReplaceWith;
};
template<typename TTarget, typename TReplaceWith>
using WithScalarType = WithScalarTypeTrait<TTarget, TReplaceWith>::Type;

template<typename TTensor>
concept HasTensorValue = requires { typename std::decay_t<TTensor>::Value; };
template<HasTensorValue... TTensors>
using ValueUnion = thes::Union<typename std::decay_t<TTensors>::Value...>;
template<HasTensorValue... TTensors>
using ScalarUnion = thes::Union<ScalarType<typename std::decay_t<TTensors>::Value>...>;

template<typename TTensor>
concept HasTensorSize = requires { typename std::decay_t<TTensor>::Size; };
template<HasTensorSize... TTensors>
using SizeIntersection = thes::Intersection<typename std::decay_t<TTensors>::Size...>;

template<typename T>
struct SimdPadSize {
  static constexpr std::size_t value = 0;
};
template<grex::Vectorizable T>
struct SimdPadSize<T> {
  static constexpr std::size_t value = grex::native_sizes<T>.back();
};
template<typename T>
inline constexpr std::size_t simd_pad_size = SimdPadSize<T>::value;
} // namespace lineal

#endif // INCLUDE_LINEAL_BASE_TYPE_TRAIT_TENSOR_HPP
