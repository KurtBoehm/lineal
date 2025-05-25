// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_COMPONENT_WISE_FACADE_DEF_HPP
#define INCLUDE_LINEAL_COMPONENT_WISE_FACADE_DEF_HPP

#include <array>
#include <cstddef>
#include <type_traits>

#include "thesauros/concepts.hpp"
#include "thesauros/types.hpp"

namespace lineal::facades {
template<typename TVec, bool tConst>
struct IteratorTrait {
  using Type = std::decay_t<TVec>::iterator;
};
template<typename TVec, bool tConst>
requires(requires {
  typename std::decay_t<TVec>::const_iterator;
} && (thes::ConstAccess<TVec> || tConst || !requires { typename std::decay_t<TVec>::iterator; }))
struct IteratorTrait<TVec, tConst> {
  using Type = std::decay_t<TVec>::const_iterator;
};

namespace detail {
template<typename TConf>
concept SupportsConstAccess =
  !requires { TConf::supports_const_access; } || TConf::supports_const_access;
template<typename TConf>
concept SupportsMutableAccess =
  requires { TConf::supports_mutable_access; } && TConf::supports_mutable_access;
template<typename TConf>
concept UsesCustomRange = requires { TConf::custom_range; } && TConf::custom_range;
template<typename TConf>
concept UsesCustomViews = requires { TConf::custom_views; } && TConf::custom_views;
template<typename TConf>
concept SupportsStore = requires { TConf::supports_store; } && TConf::supports_store;
template<typename TConf>
concept HasLocalView = !requires { TConf::has_local_view; } || TConf::has_local_view;

template<typename TConf, typename... TChildren>
inline constexpr auto component_wise_seq = [] {
  if constexpr (requires { TConf::component_wise_seq; }) {
    return TConf::component_wise_seq;
  } else {
    return thes::star::iota<0, sizeof...(TChildren)>;
  }
}();
template<typename TConf>
inline constexpr auto custom_local_seq = [] {
  if constexpr (requires { TConf::custom_local_seq; }) {
    return TConf::custom_local_seq;
  } else {
    return std::array<std::size_t, 0>{};
  }
}();

template<typename TDerived>
struct DerivedSelfGetter {
  const TDerived& operator()(const auto& v) const {
    return static_cast<const TDerived&>(v);
  }
  TDerived& operator()(auto& v) const {
    return static_cast<TDerived&>(v);
  }
};

template<bool tConst, typename TOther>
struct OtherSelfGetter {
  using Qualified = thes::ConditionalConst<tConst, TOther>;

  explicit OtherSelfGetter(Qualified& other) : other_(&other) {}

  Qualified& operator()(const auto& /*v*/) const {
    return *other_;
  }

private:
  Qualified* other_;
};
} // namespace detail
} // namespace lineal::facades

#endif // INCLUDE_LINEAL_COMPONENT_WISE_FACADE_DEF_HPP
