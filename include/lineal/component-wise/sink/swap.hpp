// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_COMPONENT_WISE_SINK_SWAP_HPP
#define INCLUDE_LINEAL_COMPONENT_WISE_SINK_SWAP_HPP

#include <cassert>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "thesauros/macropolis.hpp"
#include "thesauros/utility.hpp"

#include "lineal/base.hpp"
#include "lineal/component-wise/facade.hpp"

namespace lineal {
namespace detail {
struct SwapSinkConf {
  using Work = void;
  using Value = void;
  static constexpr bool supports_const_access = false;
  static constexpr bool supports_mutable_access = true;
};
} // namespace detail

template<AnyVector TLhs, AnyVector TRhs>
struct SwapSink
    : public facades::ComponentWiseOp<SwapSink<TLhs, TRhs>, detail::SwapSinkConf, TLhs, TRhs> {
  using Parent = facades::ComponentWiseOp<SwapSink, detail::SwapSinkConf, TLhs, TRhs>;

  using Lhs = std::decay_t<TLhs>;
  using Rhs = std::decay_t<TRhs>;
  using ValueLhs = Lhs::Value;
  using ValueRhs = Rhs::Value;
  using Size = SizeIntersection<Lhs, Rhs>;
  static constexpr bool is_shared = SharedTensors<TLhs, TRhs>;
  static constexpr auto exec_constraints = merged_exec_constraints<thes::Tuple<Lhs, Rhs>>;

  explicit SwapSink(TLhs&& lhs, TRhs&& rhs)
      : Parent(std::forward<TLhs>(lhs), std::forward<TRhs>(rhs)) {}

  THES_ALWAYS_INLINE static constexpr auto compute_iter(auto tag, const auto& /*children*/,
                                                        auto it1, auto it2) {
    const auto val1 = grex::convert_unsafe<ValueRhs>(it1.compute(tag));
    const auto val2 = grex::convert_unsafe<ValueLhs>(it2.compute(tag));
    it1.store(val2, tag);
    it2.store(val1, tag);
  }
  THES_ALWAYS_INLINE static constexpr auto
  compute_base(auto tag, const auto& arg, const auto& /*children*/, Lhs& lhs, const Rhs& rhs) {
    const auto val1 = grex::convert_unsafe<ValueRhs>(lhs.compute(arg, tag));
    const auto val2 = grex::convert_unsafe<ValueLhs>(rhs.compute(arg, tag));
    lhs.store(arg, val2, tag);
    rhs.store(arg, val1, tag);
  }

  decltype(auto) exec_instance(std::size_t /*thread_num*/) {
    // TODO Only copy the owned part if only it is valid in both?
    if constexpr (is_shared) {
      return *this;
    } else {
      auto& vec1 = lhs();
      auto& vec2 = rhs();
      vec1.validate_local_copy();
      vec2.validate_local_copy();
      using Loc1 = decltype(vec1.local_view());
      using Loc2 = decltype(vec2.local_view());
      return SwapSink<Loc1, Loc2>(vec1.local_view(), vec2.local_view());
    }
  }

  [[nodiscard]] Size size() const {
    if constexpr (is_shared) {
      return Parent::size();
    } else {
      const auto size1 = lhs().local_view().size();
      assert(size1 == rhs().local_view().size());
      return *thes::safe_cast<Size>(size1);
    }
  }

private:
  const Lhs& lhs() const {
    return thes::star::get_at<0>(this->children());
  }
  Lhs& lhs() {
    return thes::star::get_at<0>(this->children());
  }

  const Rhs& rhs() const {
    return thes::star::get_at<1>(this->children());
  }
  Rhs& rhs() {
    return thes::star::get_at<1>(this->children());
  }
};

template<AnyVector TVec1, AnyVector TVec2>
constexpr void swap(TVec1& vec1, TVec2& vec2, const auto& expo) {
  if constexpr (std::is_same_v<TVec1, TVec2>) {
    using std::swap;
    swap(vec1, vec2);
  } else {
    expo.execute(SwapSink<TVec1&, TVec2&>(vec1, vec2));
  }
}
} // namespace lineal

#endif // INCLUDE_LINEAL_COMPONENT_WISE_SINK_SWAP_HPP
