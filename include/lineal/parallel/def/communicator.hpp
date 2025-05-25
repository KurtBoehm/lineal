// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_PARALLEL_DEF_COMMUNICATOR_HPP
#define INCLUDE_LINEAL_PARALLEL_DEF_COMMUNICATOR_HPP

#include <type_traits>

namespace lineal {
struct CommunicatorBase {};
template<typename T>
struct IsCommunicatorTrait : public std::is_base_of<CommunicatorBase, T> {};
template<typename T>
concept Communicator = IsCommunicatorTrait<std::decay_t<T>>::value;
} // namespace lineal

#endif // INCLUDE_LINEAL_PARALLEL_DEF_COMMUNICATOR_HPP
