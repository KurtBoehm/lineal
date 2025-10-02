// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_COMPONENT_WISE_MATRIX_VECTOR_PRODUCT_HPP
#define INCLUDE_LINEAL_COMPONENT_WISE_MATRIX_VECTOR_PRODUCT_HPP

#include <type_traits>
#include <utility>

#include "thesauros/macropolis.hpp"
#include "thesauros/types.hpp"
#include "thesauros/utility.hpp"

#include "lineal/base.hpp"
#include "lineal/base/concept/grex.hpp"
#include "lineal/component-wise/facade.hpp"
#include "lineal/vectorization.hpp"

namespace lineal {
namespace detail {
template<typename TReal, typename TRhs>
struct MatrixVectorProductExprConf {
  using Work = void;
  using Value = WithScalarType<typename std::decay_t<TRhs>::Value, TReal>;
  static constexpr bool custom_range = true;
  static constexpr auto component_wise_seq = thes::Tuple{thes::index_tag<0>};
  static constexpr auto custom_local_seq = thes::Tuple{thes::index_tag<0>, thes::index_tag<1>};
};
} // namespace detail

template<typename TReal, AnyMatrix TLhs, AnyVector TRhs>
struct MatrixVectorProductExpr
    : public facades::ComponentWiseOp<MatrixVectorProductExpr<TReal, TLhs, TRhs>,
                                      detail::MatrixVectorProductExprConf<TReal, TRhs>, TLhs,
                                      TRhs> {
  using Conf = detail::MatrixVectorProductExprConf<TReal, TRhs>;
  using Parent =
    facades::ComponentWiseOp<MatrixVectorProductExpr<TReal, TLhs, TRhs>, Conf, TLhs, TRhs>;

  using Real = TReal;
  using Lhs = std::decay_t<TLhs>;
  using Rhs = std::decay_t<TRhs>;
  using Value = Conf::Value;
  using Size = SizeIntersection<Lhs, Rhs>;
  static constexpr bool is_shared = SharedTensors<TLhs, TRhs>;

  explicit MatrixVectorProductExpr(TLhs&& lhs, TRhs&& rhs)
      : Parent(std::forward<TLhs>(lhs), std::forward<TRhs>(rhs)) {}

  template<TensorTag<Value> TTag>
  THES_ALWAYS_INLINE auto compute_impl(TTag tag, const auto& children, auto get_row,
                                       auto get_row_off) const {
    const auto& rhs = thes::star::get_at<1>(children);

    if constexpr (grex::AnyVectorTag<TTag> && grex::is_geometry_respecting<TTag> &&
                  BandedIterable<Lhs, TTag>) {
      decltype(auto) row = get_row();

      auto sum = grex::zeros<Real>(tag);
      auto rhs_it = rhs.begin() + row.index();
      row.banded_iterate(
        [&](auto val, auto off) THES_ALWAYS_INLINE {
          sum +=
            grex::convert_unsafe<Real>(val) * grex::convert_unsafe<Real>(off(rhs_it).compute(tag));
        },
        unordered_tag, tag);
      return sum;
    } else {
      return grex::transform(
        [&](auto i) THES_ALWAYS_INLINE {
          Value sum = compat::zero<Value>();
          get_row_off(i).iterate(
            [&](auto j, auto val) THES_ALWAYS_INLINE {
              sum = compat::fmadd(compat::cast<Real>(val), compat::cast<Real>(rhs[j]), sum);
            },
            valued_tag, unordered_tag);
          return sum;
        },
        tag);
    }
  }

  THES_ALWAYS_INLINE auto compute_iter(TensorTag<Value> auto tag, const auto& children,
                                       auto lhs_it) const {
    return compute_impl(
      tag, children, [&]() THES_ALWAYS_INLINE { return *lhs_it; },
      [&](auto off) THES_ALWAYS_INLINE { return lhs_it[off]; });
  }
  THES_ALWAYS_INLINE auto compute_base(TensorTag<Value> auto tag, const auto& arg,
                                       const auto& children, const auto& lhs) const {
    return compute_impl(
      tag, children, [&]() THES_ALWAYS_INLINE { return lhs[arg]; },
      [&](auto off) THES_ALWAYS_INLINE { return lhs[arg + Size{off}]; });
  }

  THES_ALWAYS_INLINE auto begin_impl(thes::IndexTag<0> /*tag*/, const auto& lhs) const {
    return std::begin(lhs);
  }
  THES_ALWAYS_INLINE auto end_impl(thes::IndexTag<0> /*tag*/, const auto& lhs) const {
    return std::end(lhs);
  }

  THES_ALWAYS_INLINE Size size() const {
    if constexpr (is_shared) {
      return lhs().row_num();
    } else {
      return lhs().own_row_num();
    }
  }
  THES_ALWAYS_INLINE Size own_size_impl() const {
    return size();
  }

  static decltype(auto) child_own_view(thes::IndexTag<0> /*tag*/, TLhs& child) {
    return child.own_rows_view();
  }
  static decltype(auto) child_own_view(thes::IndexTag<1> /*tag*/, TRhs& child) {
    child.validate_local_copy();
    return child.local_view();
  }

private:
  const auto& lhs() const {
    return thes::star::get_at<0>(this->children());
  }
};

template<typename TReal, AnyMatrix TLhs, AnyVector TRhs>
constexpr auto multiply(TLhs&& lhs, TRhs&& rhs) {
  return MatrixVectorProductExpr<TReal, TLhs, TRhs>(std::forward<TLhs>(lhs),
                                                    std::forward<TRhs>(rhs));
}
template<AnyMatrix TLhs, AnyVector TRhs>
constexpr auto operator*(TLhs&& lhs, TRhs&& rhs) {
  using Real = ScalarUnion<TLhs, TRhs>;
  return multiply<Real>(std::forward<TLhs>(lhs), std::forward<TRhs>(rhs));
}
} // namespace lineal

#endif // INCLUDE_LINEAL_COMPONENT_WISE_MATRIX_VECTOR_PRODUCT_HPP
