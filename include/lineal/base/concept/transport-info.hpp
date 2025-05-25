// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_BASE_CONCEPT_TRANSPORT_INFO_HPP
#define INCLUDE_LINEAL_BASE_CONCEPT_TRANSPORT_INFO_HPP

#include <type_traits>
#include <variant>

namespace lineal {
struct SharedTransportInfoBase {};
template<typename T>
struct IsSharedTransportInfoTrait : public std::is_base_of<SharedTransportInfoBase, T> {};
template<typename T>
concept SharedTransportInfo = IsSharedTransportInfoTrait<std::decay_t<T>>::value;

struct DistributedTransportInfoBase {};
template<typename T>
struct IsDistributedTransportInfoTrait : public std::is_base_of<DistributedTransportInfoBase, T> {};
template<typename T>
concept DistributedTransportInfo = IsDistributedTransportInfoTrait<std::decay_t<T>>::value;

template<typename T>
concept TransportInfo = SharedTransportInfo<T> || DistributedTransportInfo<T>;

template<typename T>
struct TransportInfoVariantTrait : public std::false_type {};
template<TransportInfo... Ts>
struct TransportInfoVariantTrait<std::variant<Ts...>> : public std::true_type {};

template<typename T>
concept TransportInfos = TransportInfo<T> || TransportInfoVariantTrait<std::decay_t<T>>::value;
} // namespace lineal

#endif // INCLUDE_LINEAL_BASE_CONCEPT_TRANSPORT_INFO_HPP
