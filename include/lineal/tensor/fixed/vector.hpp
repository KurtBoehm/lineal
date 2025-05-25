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

#include "thesauros/static-ranges.hpp"
#include "thesauros/types.hpp"

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
  template<AnyVector TVec>
  requires(TVec::size == size)
  explicit constexpr DenseVector(const TVec& other)
      : data_{thes::star::index_transform<size>([&](auto i) { return other[i]; }) |
              thes::star::to_array} {}

  template<std::size_t tIdx>
  constexpr Value operator[](thes::AutoTag<tIdx> /*idx*/) const {
    return std::get<tIdx>(data_);
  }
  template<std::size_t tIdx>
  constexpr Value& operator[](thes::AutoTag<tIdx> /*idx*/) {
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

  template<typename TOther>
  constexpr auto cast() const {
    return DenseVector<TOther, tSize>{
      data_ | thes::star::transform([](Value v) { return static_cast<TOther>(v); }) |
      thes::star::to_array};
  }

  constexpr bool operator==(const DenseVector& other) const = default;

private:
  template<typename TOther, std::size_t tOtherDim>
  friend struct DenseVector;
  explicit constexpr DenseVector(Data&& data) : data_(std::move(data)) {}

  Data data_;
};
} // namespace lineal::fix

#endif // INCLUDE_LINEAL_TENSOR_FIXED_VECTOR_HPP
