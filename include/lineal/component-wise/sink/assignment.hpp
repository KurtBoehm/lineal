// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_COMPONENT_WISE_SINK_ASSIGNMENT_HPP
#define INCLUDE_LINEAL_COMPONENT_WISE_SINK_ASSIGNMENT_HPP

#include <cassert>
#include <type_traits>
#include <utility>

#include "thesauros/macropolis.hpp"

#include "lineal/base.hpp"
#include "lineal/component-wise/facade.hpp"
#include "lineal/parallel.hpp"

namespace lineal {
namespace detail {
template<typename TValue, bool tDist>
struct AssignSinkConf {
  using Work = void;
  using Value = TValue;
  static constexpr bool custom_range = tDist;
  static constexpr bool custom_views = tDist;
  static constexpr bool supports_const_access = false;
  static constexpr bool supports_mutable_access = true;
};

template<typename TDst, typename TSrc, bool tForceOwn>
struct AssignTagTrait {
  using Dst = std::decay_t<TDst>;
  using Src = std::decay_t<TSrc>;
  static constexpr bool use_local_view = !tForceOwn && HaveLocalView<Dst, Src>;
  using Tag = std::conditional_t<use_local_view, LocalIndexTag, OwnIndexTag>;
};
template<typename TDst, typename TSrc, bool tForceOwn = false>
using AssignTag = AssignTagTrait<TDst, TSrc, tForceOwn>::Tag;
} // namespace detail

template<typename TChild, AnyVector TDst, AnyVector TSrc, OptIndexTag TIdxTag = void>
struct AssignSinkBase
    : public facades::ComponentWiseOp<
        TChild,
        detail::AssignSinkConf<typename std::decay_t<TSrc>::Value, !std::is_void_v<TIdxTag>>, TDst,
        TSrc> {
  static constexpr bool is_shared = std::is_void_v<TIdxTag>;

  using Dst = std::decay_t<TDst>;
  using Src = std::decay_t<TSrc>;

  using DstValue = Dst::Value;
  using Value = Src::Value;

  using Parent =
    facades::ComponentWiseOp<TChild, detail::AssignSinkConf<Value, !is_shared>, TDst, TSrc>;
  using Size = SizeIntersection<Dst, Src>;

  AssignSinkBase(TDst&& dst, TSrc&& src)
      : Parent(std::forward<TDst>(dst), std::forward<TSrc>(src)) {}

  THES_ALWAYS_INLINE static constexpr auto compute_iter(auto tag, [[maybe_unused]] auto& children,
                                                        auto dst_it, auto src_it)
  requires(requires() { dst_it.store(grex::convert_unsafe<DstValue>(src_it.compute(tag)), tag); })
  {
    assert(src_it - thes::star::get_at<1>(children).begin() <
           thes::star::get_at<1>(children).size());
    const auto val = src_it.compute(tag);
    dst_it.store(grex::convert_unsafe<DstValue>(val), tag);
    return val;
  }
  THES_ALWAYS_INLINE static constexpr auto
  compute_base(auto tag, const auto& arg, const auto& /*children*/, auto& dst, auto& src)
  requires(requires() {
    dst.store(add_tag<TIdxTag>(arg),
              grex::convert_unsafe<DstValue>(src.compute(add_tag<TIdxTag>(arg), tag)), tag);
  })
  {
    const auto arg_tag = add_tag<TIdxTag>(arg);
    const auto val = src.compute(arg_tag, tag);
    dst.store(arg_tag, grex::convert_unsafe<DstValue>(val), tag);
    return val;
  }

protected:
  [[nodiscard]] const Dst& dst() const {
    return thes::star::get_at<0>(this->children());
  }
  [[nodiscard]] Dst& dst() {
    return thes::star::get_at<0>(this->children());
  }

  [[nodiscard]] const TSrc& src() const {
    return thes::star::get_at<1>(this->children());
  }
  [[nodiscard]] TSrc& src() {
    return thes::star::get_at<1>(this->children());
  }
};

template<typename TDst, typename TSrc>
struct AssignSink;

template<SharedVector TDst, SharedVector TSrc>
struct AssignSink<TDst, TSrc> : public AssignSinkBase<AssignSink<TDst, TSrc>, TDst, TSrc> {
  using Parent = AssignSinkBase<AssignSink, TDst, TSrc>;
  using Size = Parent::Size;
  using Parent::Parent;
};

template<AnyVector TDst, AnyVector TSrc>
constexpr void assign(TDst&& dst, TSrc&& src, const auto& expo) {
  expo.execute(AssignSink<TDst, TSrc>(std::forward<TDst>(dst), std::forward<TSrc>(src)));
}

template<AnyVector TDst, AnyVector TSrc>
constexpr auto assign_expr(TDst&& dst, TSrc&& src) {
  return AssignSink<TDst, TSrc>(std::forward<TDst>(dst), std::forward<TSrc>(src));
}

template<SharedVector TDst, SharedVector TSrc>
constexpr TDst create_from(TSrc&& src, const auto& expo) {
  TDst dst(src.size());
  assign(dst, std::forward<TSrc>(src), expo);
  return dst;
}
} // namespace lineal

#endif // INCLUDE_LINEAL_COMPONENT_WISE_SINK_ASSIGNMENT_HPP
