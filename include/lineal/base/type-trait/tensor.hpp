// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_BASE_TYPE_TRAIT_TENSOR_HPP
#define INCLUDE_LINEAL_BASE_TYPE_TRAIT_TENSOR_HPP

#include <type_traits>

#include "thesauros/types.hpp"

namespace lineal {
template<typename TTensor>
concept HasTensorValue = requires { typename std::decay_t<TTensor>::Value; };
template<HasTensorValue... TTensors>
using ValueUnion = thes::Union<typename std::decay_t<TTensors>::Value...>;

template<typename TTensor>
concept HasTensorSize = requires { typename std::decay_t<TTensor>::Size; };
template<HasTensorSize... TTensors>
using SizeIntersection = thes::Intersection<typename std::decay_t<TTensors>::Size...>;
} // namespace lineal

#endif // INCLUDE_LINEAL_BASE_TYPE_TRAIT_TENSOR_HPP
