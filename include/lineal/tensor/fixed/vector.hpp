// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_TENSOR_FIXED_VECTOR_HPP
#define INCLUDE_LINEAL_TENSOR_FIXED_VECTOR_HPP

#include <array>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <type_traits>

#include "thesauros/static-ranges.hpp"
#include "thesauros/types.hpp"
#include "thesauros/utility.hpp"

#include "lineal/base/compat/operation.hpp"
#include "lineal/base/concept.hpp"
#include "lineal/base/type-trait/tensor.hpp"
#include "lineal/tensor/fixed/concepts.hpp"

namespace lineal::fix {
template<typename TValue, std::size_t tSize>
struct DenseVector : public VectorBase {
  using Value = TValue;
  static constexpr std::size_t size = tSize;
  using Data = std::array<Value, size>;

  static constexpr DenseVector zero() {
    return DenseVector(thes::star::constant<size>(TValue{0}) | thes::star::to_array);
  }

  constexpr DenseVector() = default;
  explicit constexpr DenseVector(std::initializer_list<Value> init)
      : data_{thes::star::index_transform<size>([&](auto i) { return init.begin()[i]; }) |
              thes::star::to_array} {
    assert(init.size() == size);
  }
  explicit constexpr DenseVector(Data&& data) : data_(std::move(data)) {}

  template<AnyVector TVec>
  requires(TVec::size == size)
  explicit constexpr DenseVector(const TVec& other)
      : data_{thes::star::index_transform<size>([&](auto i) { return other[i]; }) |
              thes::star::to_array} {}
  explicit constexpr DenseVector(TValue value)
      : data_{thes::star::constant<size>(value) | thes::star::to_array} {}

  template<std::size_t tIdx>
  constexpr Value operator[](thes::IndexTag<tIdx> /*idx*/) const {
    return std::get<tIdx>(data_);
  }
  template<std::size_t tIdx>
  constexpr Value& operator[](thes::IndexTag<tIdx> /*idx*/) {
    return std::get<tIdx>(data_);
  }

  constexpr Value operator[](std::size_t idx) const {
    return data_[idx];
  }
  constexpr Value& operator[](std::size_t idx) {
    return data_[idx];
  }

  const Value* data() const {
    return data_.data();
  }
  Value* data() {
    return data_.data();
  }

  const Value* begin() const {
    return data_.data();
  }
  Value* begin() {
    return data_.data();
  }
  const Value* end() const {
    return data_.data() + data_.size();
  }
  Value* end() {
    return data_.data() + data_.size();
  }

  constexpr bool operator==(const DenseVector& other) const = default;

  // TODO This is public to make `DenseVector` structural, but if C++ ever allows instances of types
  //      with private members to be used as template parameters, this should really be private
  Data data_;

private:
  template<typename TOther, std::size_t tOtherDim>
  friend struct DenseVector;
};

template<typename TFun, AnyVector... TInner>
struct TransformedVectorView : public VectorBase {
  using Inner = thes::Tuple<std::decay_t<TInner>...>;
  using Value =
    decltype(std::declval<TFun>()(std::declval<typename std::decay_t<TInner>::Value>()...));
  static constexpr std::size_t size =
    thes::star::unique_value(std::array{std::decay_t<TInner>::size...}).value();

  explicit constexpr TransformedVectorView(TFun&& fun, TInner&&... inner)
      : fun_(std::forward<TFun>(fun)), inner_{std::forward<TInner>(inner)...} {}

  template<std::size_t tIdx>
  requires(tIdx < size)
  constexpr Value operator[](thes::IndexTag<tIdx> idx) const {
    return thes::star::static_apply<sizeof...(TInner)>(
      [&]<std::size_t... tI>() { return fun_(get<tI>(inner_)[idx]...); });
  }
  constexpr Value operator[](std::size_t idx) const {
    return thes::star::static_apply<sizeof...(TInner)>(
      [&]<std::size_t... tI>() { return fun_(get<tI>(inner_)[idx]...); });
  }

private:
  [[no_unique_address]] TFun fun_;
  thes::Tuple<TInner...> inner_;
};

template<typename TFun, AnyVector... TVec>
constexpr auto transform(TFun&& fun, TVec&&... vec) {
  return TransformedVectorView<TFun, TVec...>{std::forward<TFun>(fun), std::forward<TVec>(vec)...};
}

#define LINEAL_FIX_DVEC_ARITH(OP) \
  template<AnyVector TLhs, AnyVector TRhs> \
  requires(TLhs::size == TRhs::size) \
  constexpr auto operator OP(const TLhs& lhs, const TRhs& rhs) { \
    using Value = ValueUnion<TLhs, TRhs>; \
    constexpr std::size_t size = std::decay_t<TLhs>::size; \
    return DenseVector<Value, size>{thes::star::static_apply<size>([&]<std::size_t... tI>() { \
      return std::array{Value(lhs[thes::index_tag<tI>] OP rhs[thes::index_tag<tI>])...}; \
    })}; \
  } \
  template<AnyVector TLhs, AnyVector TRhs> \
  requires(TLhs::size == TRhs::size) \
  constexpr TLhs& operator OP##=(TLhs & lhs, const TRhs & rhs) { \
    constexpr std::size_t size = std::decay_t<TLhs>::size; \
    thes::star::static_apply<size>([&]<std::size_t... tI>() { \
      (..., (lhs[thes::index_tag<tI>] OP## = rhs[thes::index_tag<tI>])); \
    }); \
    return lhs; \
  }

LINEAL_FIX_DVEC_ARITH(+)
LINEAL_FIX_DVEC_ARITH(-)
#undef LINEAL_FIX_DVEC_ARITH

template<AnyVector TLhs, IsScalar TRhs>
constexpr auto operator/(TLhs&& lhs, TRhs rhs) {
  return transform([rhs](auto a) { return a / rhs; }, std::forward<TLhs>(lhs));
}
template<IsScalar TLhs, AnyVector TRhs>
constexpr auto operator*(TLhs lhs, TRhs&& rhs) {
  return transform([lhs](auto val) { return lhs * val; }, std::forward<TRhs>(rhs));
}
template<AnyVector TLhs, IsScalar TRhs>
constexpr auto operator*(TLhs&& lhs, TRhs rhs) {
  return transform([rhs](auto val) { return val * rhs; }, std::forward<TLhs>(lhs));
}

template<typename TOther, AnyVector TVec>
constexpr auto cast(TVec&& vec) {
  return transform([](auto val) { return TOther(val); }, std::forward<TVec>(vec));
}

template<typename TReal, AnyVector TLhs, AnyVector TRhs>
requires(TLhs::size == TRhs::size)
[[nodiscard]] constexpr TReal dot(const TLhs& lhs, const TRhs& rhs) {
  return thes::star::static_apply<TLhs::size>([&]<std::size_t... tI> {
    return (TReal{0} + ... + (lhs[thes::index_tag<tI>] * rhs[thes::index_tag<tI>]));
  });
}

template<typename TReal, AnyVector TVec>
[[nodiscard]] constexpr TReal euclidean_squared(const TVec& vec) {
  return thes::star::static_apply<TVec::size>([&]<std::size_t... tI> {
    return (TReal{0} + ... + (vec[thes::index_tag<tI>] * vec[thes::index_tag<tI>]));
  });
}
template<typename TReal, AnyVector TVec>
[[nodiscard]] constexpr TReal euclidean_norm(const TVec& vec) {
  return compat::sqrt(euclidean_squared<TReal>(vec));
}

template<typename TReal, AnyVector TVec>
[[nodiscard]] constexpr TReal max_norm(const TVec& vec) {
  TReal output = 0;
  thes::star::static_apply<TVec::size>([&]<std::size_t... tI> {
    (..., (output = std::max(output, std::abs(vec[thes::index_tag<tI>]))));
  });
  return output;
}
} // namespace lineal::fix

#endif // INCLUDE_LINEAL_TENSOR_FIXED_VECTOR_HPP
