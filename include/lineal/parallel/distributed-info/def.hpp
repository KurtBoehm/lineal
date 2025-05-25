// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_PARALLEL_DISTRIBUTED_INFO_DEF_HPP
#define INCLUDE_LINEAL_PARALLEL_DISTRIBUTED_INFO_DEF_HPP

#include <type_traits>

namespace lineal {
template<typename T>
concept OptDistributedInfo = std::is_void_v<T>;

template<typename TDefs>
struct DistributedInfoOfTrait {
  using Type = void;
};
template<typename TDefs>
requires(requires { typename TDefs::DistributedInfo; })
struct DistributedInfoOfTrait<TDefs> {
  using Type = TDefs::DistributedInfo;
};
template<typename TDefs>
using DistributedInfoOf = DistributedInfoOfTrait<TDefs>::Type;
template<typename T>
concept HasDistributedInfo = requires(const T& v) { v.distributed_info(); };
template<typename... Ts>
concept HaveDistributedInfo = (... && HasDistributedInfo<Ts>);

template<typename TDistInfo, typename TFallback>
struct DistributedSizesTrait {
  using DistInfo = std::decay_t<TDistInfo>;
  using LocalSize = DistInfo::LocalSize;
  using GlobalSize = DistInfo::GlobalSize;
};
template<typename TFallback>
struct DistributedSizesTrait<void, TFallback> {
  using LocalSize = TFallback;
  using GlobalSize = TFallback;
};
template<typename TDistInfo, typename TFallback>
using LocalSizeOf = DistributedSizesTrait<TDistInfo, TFallback>::LocalSize;
template<typename TDistInfo, typename TFallback>
using GlobalSizeOf = DistributedSizesTrait<TDistInfo, TFallback>::GlobalSize;
} // namespace lineal

#endif // INCLUDE_LINEAL_PARALLEL_DISTRIBUTED_INFO_DEF_HPP
