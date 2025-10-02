// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_COMPONENT_WISE_VECTOR_EXPRESSION_HPP
#define INCLUDE_LINEAL_COMPONENT_WISE_VECTOR_EXPRESSION_HPP

#include <cassert>
#include <concepts>
#include <functional>
#include <span>
#include <type_traits>
#include <utility>

#include "thesauros/algorithms.hpp"
#include "thesauros/concepts.hpp"
#include "thesauros/macropolis.hpp"
#include "thesauros/ranges.hpp"
#include "thesauros/types.hpp"

#include "lineal/base.hpp"
#include "lineal/component-wise/facade.hpp"
#include "lineal/parallel.hpp"
#include "lineal/vectorization.hpp"

namespace lineal {
namespace detail {
template<typename TReal, AnyVector... TVecs>
struct OpExprConf {
  using RealValue =
    thes::TypeSeq<WithScalarType<typename std::decay_t<TVecs>::Value, TReal>...>::Unique;

  using Work = TReal;
  using Value = RealValue;
};
} // namespace detail

template<typename TReal, typename TOp, AnyVector... TVecs>
struct OpExpr : public facades::ComponentWiseOp<OpExpr<TReal, TOp, TVecs...>,
                                                detail::OpExprConf<TReal, TVecs...>, TVecs...> {
  using Parent = facades::ComponentWiseOp<OpExpr, detail::OpExprConf<TReal, TVecs...>, TVecs...>;

  explicit OpExpr(TOp&& op, TVecs&&... vecs)
      : Parent(std::forward<TVecs>(vecs)...), op_(std::move(op)) {}

  THES_ALWAYS_INLINE constexpr auto compute_val(auto tag, const auto& /*children*/,
                                                auto... args) const {
    if constexpr (requires { op_(args..., tag); }) {
      return op_(args..., tag);
    } else {
      return op_(args...);
    }
  }

private:
  TOp op_;
};
namespace detail {
template<typename TReal, typename TOp, AnyVector... TVecs>
inline OpExpr<TReal, TOp, TVecs...> make_op_expr(TOp op, TVecs&&... vecs) {
  return OpExpr<TReal, TOp, TVecs...>(std::move(op), std::forward<TVecs>(vecs)...);
}
} // namespace detail

template<typename TReal, AnyVector TVec1, AnyVector TVec2>
constexpr auto add(TVec1&& v1, TVec2&& v2) {
  return detail::make_op_expr<TReal>(std::plus{}, std::forward<TVec1>(v1), std::forward<TVec2>(v2));
}
template<typename TReal, AnyVector TVec1, AnyVector TVec2>
constexpr auto subtract(TVec1&& v1, TVec2&& v2) {
  return detail::make_op_expr<TReal>(std::minus{}, std::forward<TVec1>(v1),
                                     std::forward<TVec2>(v2));
}
template<typename TReal, AnyVector TVec1, AnyVector TVec2>
constexpr auto cw_multiply(TVec1&& v1, TVec2&& v2) {
  return detail::make_op_expr<TReal>(std::multiplies{}, std::forward<TVec1>(v1),
                                     std::forward<TVec2>(v2));
}
template<typename TReal, AnyVector TVec1, AnyVector TVec2>
constexpr auto cw_divide(TVec1&& v1, TVec2&& v2) {
  return detail::make_op_expr<TReal>(std::divides{}, std::forward<TVec1>(v1),
                                     std::forward<TVec2>(v2));
}

template<IsScalar TReal, AnyVector TVec>
constexpr auto scale(TVec&& vector, TReal scalar) {
  return detail::make_op_expr<TReal>(
    [scalar](auto value) { return compat::cast<TReal>(scalar * compat::cast<TReal>(value)); },
    std::forward<TVec>(vector));
}
template<AnyVector TVec>
constexpr auto cw_abs(TVec&& vector) {
  using Value = std::decay_t<TVec>::Value;
  return detail::make_op_expr<Value>([]<typename TValue>(TValue value) { return grex::abs(value); },
                                     std::forward<TVec>(vector));
}

namespace detail {
template<typename TReal, typename TSize>
struct ConstantExprConf {
  using Value = TReal;
  using Size = TSize;
  using IndexTag = LocalIndexTag;
};
} // namespace detail

template<typename TReal>
struct ConstantExprBase {
  explicit ConstantExprBase(TReal value) : value_(value) {}

  THES_ALWAYS_INLINE constexpr auto compute_impl(auto tag, auto /*idx*/) const
  requires(requires(TReal value) { compat::broadcast(value, tag); })
  {
    return compat::broadcast(value_, tag);
  }
  decltype(auto) lookup(auto /*idxs*/, grex::AnyTag auto tag) const
  requires(requires(TReal value) { compat::broadcast(value, tag); })
  {
    return compat::broadcast(value_, tag);
  }

private:
  TReal value_;
};

template<typename TReal, typename TSize>
struct SharedConstantExpr
    : public ConstantExprBase<TReal>,
      public facades::SharedNullaryCwOp<SharedConstantExpr<TReal, TSize>,
                                        detail::ConstantExprConf<TReal, TSize>> {
  using Base = ConstantExprBase<TReal>;
  using Facade =
    facades::SharedNullaryCwOp<SharedConstantExpr, detail::ConstantExprConf<TReal, TSize>>;

  explicit SharedConstantExpr(TSize size, TReal value) : Base(value), Facade(size) {}
};

template<std::unsigned_integral TSize, typename TReal>
constexpr auto constant(TSize size, TReal value) {
  return SharedConstantExpr<TReal, TSize>(size, value);
}

constexpr auto constant_like(const SharedVector auto& src, auto value) {
  return constant(src.size(), value);
}

template<AnyVector TVec1, AnyVector TVec2>
constexpr auto operator+(TVec1&& v1, TVec2&& v2) {
  using Real = ScalarUnion<TVec1, TVec2>;
  return add<Real>(std::forward<TVec1>(v1), std::forward<TVec2>(v2));
}
template<AnyVector TVec1, AnyVector TVec2>
constexpr auto operator-(TVec1&& v1, TVec2&& v2) {
  using Real = ScalarUnion<TVec1, TVec2>;
  return subtract<Real>(std::forward<TVec1>(v1), std::forward<TVec2>(v2));
}

template<AnyVector TVec, IsScalar TReal>
constexpr auto operator*(TVec&& vector, TReal scalar) {
  return scale(std::forward<TVec>(vector), scalar);
}
template<IsScalar TReal, AnyVector TVec>
constexpr auto operator*(TReal scalar, TVec&& vector) {
  return scale(std::forward<TVec>(vector), scalar);
}

namespace detail {
template<typename TVec>
struct SubVectorExprConf {
  using Work = void;
  using Value = std::decay_t<TVec>::Value;
  static constexpr bool custom_range = true;
  static constexpr bool is_const = thes::ConstAccess<TVec>;
  static constexpr bool supports_mutable_access = !is_const;
  static constexpr bool supports_store = !is_const;
};
} // namespace detail

template<typename TRange, SharedVector TVec>
struct SubVectorExpr;
template<typename TSize, SharedVector TVec>
struct SubVectorExpr<thes::IotaRange<TSize>, TVec>
    : public facades::ComponentWiseOp<SubVectorExpr<thes::IotaRange<TSize>, TVec>,
                                      detail::SubVectorExprConf<TVec>, TVec> {
  using Range = thes::IotaRange<TSize>;
  using Vec = std::decay_t<TVec>;
  using Value = Vec::Value;
  using Size = Vec::Size;
  using GlobalSize = Vec::GlobalSize;
  using Index = Vec::Index;
  static constexpr bool is_const = thes::ConstAccess<TVec>;
  static constexpr bool is_shared = Vec::is_shared;

  using Parent = facades::ComponentWiseOp<SubVectorExpr, detail::SubVectorExprConf<TVec>, TVec>;

  explicit SubVectorExpr(Range range, TVec&& vec)
      : Parent(std::forward<TVec>(vec)), begin_(range.begin_value()), end_(range.end_value()),
        size_(range.size()) {
    if constexpr (!is_shared) {
      assert((range == distributed_info().index_range(local_index_tag)) ||
             (range == distributed_info().index_range_within(own_index_tag, local_index_tag)));
    }
  }

  THES_ALWAYS_INLINE constexpr auto begin_impl(thes::IndexTag<0> /*tag*/, TVec& vec) const {
    return std::begin(vec) + begin_;
  }
  THES_ALWAYS_INLINE constexpr auto end_impl(thes::IndexTag<0> /*tag*/, TVec& vec) const {
    return std::begin(vec) + end_;
  }
  THES_ALWAYS_INLINE constexpr Size size() const {
    return size_;
  }

  THES_ALWAYS_INLINE static constexpr decltype(auto)
  compute_iter(auto tag, [[maybe_unused]] const auto& children, auto it) {
    assert(typename std::decay_t<decltype(thes::star::get_at<0>(children))>::const_iterator{it} -
             std::as_const(thes::star::get_at<0>(children)).begin() <
           thes::star::get_at<0>(children).size());
    return it.compute(tag);
  }
  THES_ALWAYS_INLINE constexpr decltype(auto)
  compute_base(auto tag, TypedIndex<is_shared, Size, GlobalSize> auto idx, const auto& /*children*/,
               TVec& vec) const {
    return vec.compute(adjust_idx(idx), tag);
  }
  THES_ALWAYS_INLINE constexpr decltype(auto) compute_base(auto tag,
                                                           thes::AnyIndexPosition auto idx,
                                                           const auto& /*children*/,
                                                           TVec& vec) const {
    return vec.compute(adjust_idx(idx), tag);
  }

  THES_ALWAYS_INLINE static constexpr void store_impl(auto tag, auto value,
                                                      const auto& /*children*/, auto it) {
    return it.store(value, tag);
  }
  THES_ALWAYS_INLINE constexpr void store_impl(auto tag,
                                               TypedIndex<is_shared, Size, GlobalSize> auto idx,
                                               auto value, const auto& /*children*/,
                                               TVec& vec) const {
    return vec.store(adjust_idx(idx), value, tag);
  }
  THES_ALWAYS_INLINE constexpr void store_impl(auto tag, thes::AnyIndexPosition auto idx,
                                               auto value, const auto& /*children*/,
                                               TVec& vec) const {
    return vec.store(adjust_idx(idx), value, tag);
  }

  const Value* data() const {
    return child().data() + begin_;
  }
  Value* data()
  requires(!is_const)
  {
    return child().data() + begin_;
  }

  [[nodiscard]] std::span<const Value> span() const {
    return std::span{data(), size()};
  }
  [[nodiscard]] std::span<Value> span()
  requires(!is_const)
  {
    return std::span{data(), size()};
  }

  [[nodiscard]] std::span<const Value> span(thes::IotaRange<Size> range) const {
    return child().span(thes::range(range.begin_value() + begin_, range.end_value() + begin_));
  }
  [[nodiscard]] std::span<Value> span(thes::IotaRange<Size> range)
  requires(!is_const)
  {
    return child().span(thes::range(range.begin_value() + begin_, range.end_value() + begin_));
  }

  const auto& distributed_info() const
  requires(requires { this->child().distributed_info(); })
  {
    return child().distributed_info();
  }
  const auto& communicator() const {
    return child().communicator();
  }

private:
  const Vec& child() const {
    return thes::star::get_at<0>(this->children());
  }
  Vec& child() {
    return thes::star::get_at<0>(this->children());
  }

  auto adjust_idx(TypedIndex<is_shared, Size, GlobalSize> auto idx) const {
    if constexpr (is_shared) {
      return idx + begin_;
    } else {
      return idx;
    }
  }
  auto adjust_idx(thes::AnyIndexPosition auto idx) const {
    static_assert(TypedIndex<decltype(idx.index), is_shared, Size, GlobalSize>);
    if constexpr (is_shared) {
      return idx.index + begin_;
    } else {
      return idx;
    }
  }

  Size begin_;
  Size end_;
  Size size_;
};

template<typename TRange, SharedVector TVec>
constexpr auto sub_vector(TRange&& range, TVec&& vec) {
  return SubVectorExpr<TRange, TVec>(std::forward<TRange>(range), std::forward<TVec>(vec));
}
} // namespace lineal

#endif // INCLUDE_LINEAL_COMPONENT_WISE_VECTOR_EXPRESSION_HPP
