// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_BASE_CONCEPT_RANGE_HPP
#define INCLUDE_LINEAL_BASE_CONCEPT_RANGE_HPP

#include <type_traits>

#include "lineal/base/tag.hpp"

namespace lineal {
// Vector

struct SharedVectorBase {};
template<typename T>
struct IsSharedVectorTrait : public std::is_base_of<SharedVectorBase, T> {};
template<typename T>
concept SharedVector = IsSharedVectorTrait<std::decay_t<T>>::value;

template<typename T>
concept AnyVector = SharedVector<T>;

// Matrix

struct SharedMatrixBase {};
template<typename T>
struct IsSharedMatrixTrait : public std::is_base_of<SharedMatrixBase, T> {};
template<typename T>
concept SharedMatrix = IsSharedMatrixTrait<std::decay_t<T>>::value;

template<typename T>
concept AnyMatrix = SharedMatrix<T>;

template<typename TMat, typename TTag>
concept BandedIterable = requires(const std::decay_t<TMat>::ConstRow& row, TTag tag) {
  row.banded_iterate([](auto, auto) {}, unordered_tag, tag);
};

// Tensor

template<typename T>
concept SharedTensor = SharedVector<T> || SharedMatrix<T>;

template<typename T>
concept AnyTensor = AnyVector<T> || AnyMatrix<T>;

template<typename... TTensors>
struct SharedTensorsTrait;
template<SharedTensor... TTensors>
struct SharedTensorsTrait<TTensors...> : public std::true_type {};

template<typename... TTensors>
concept SharedTensors = SharedTensorsTrait<TTensors...>::value;

// Graphs

struct SharedGraphBase {};
template<typename T>
struct IsSharedGraphTrait : public std::is_base_of<SharedGraphBase, T> {};
template<typename T>
concept SharedGraph = IsSharedGraphTrait<std::decay_t<T>>::value;

template<typename T>
concept AnyGraph = SharedGraph<T>;

// Ranges

template<typename T>
concept SharedRange = SharedTensor<T> || SharedGraph<T>;
template<typename... TRanges>
concept SharedRanges = (... && SharedRange<TRanges>);

template<typename T>
concept AnyRange = AnyTensor<T> || AnyGraph<T>;
} // namespace lineal

#endif // INCLUDE_LINEAL_BASE_CONCEPT_RANGE_HPP
