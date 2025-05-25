// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_BASE_CONCEPT_VALUATOR_HPP
#define INCLUDE_LINEAL_BASE_CONCEPT_VALUATOR_HPP

#include <type_traits>

namespace lineal {
struct ValuatorBase {};
template<typename T>
struct IsValuatorTrait : public std::is_base_of<ValuatorBase, T> {};
template<typename T>
concept Valuator = IsValuatorTrait<std::decay_t<T>>::value;
} // namespace lineal

#endif // INCLUDE_LINEAL_BASE_CONCEPT_VALUATOR_HPP
