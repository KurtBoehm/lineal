// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_TENSOR_DYNAMIC_STENCIL_STORAGE_CELL_STORAGE_HPP
#define INCLUDE_LINEAL_TENSOR_DYNAMIC_STENCIL_STORAGE_CELL_STORAGE_HPP

#include <cstddef>
#include <memory>
#include <span>

#include "thesauros/containers.hpp"
#include "thesauros/memory.hpp"

#include "lineal/vectorization.hpp"

namespace lineal {
template<typename T, typename TAlloc = thes::HugePagesAllocator<T>>
struct CellStorage {
  using Value = T;
  using Size = std::size_t;
  using Alloc = TAlloc;
  static constexpr std::size_t max_vector_size = grex::register_bytes.back();

  using Data = thes::array::TypedChunk<Value, Size, Alloc>;
  using iterator = Value*;
  using const_iterator = const Value*;

  explicit CellStorage(Size size) : values_(size + max_vector_size + 2), size_(size) {
    init_padding();
    std::uninitialized_default_construct(begin(), end());
  }

  explicit CellStorage(Size size, Value value) : values_(size + max_vector_size + 2), size_(size) {
    init_padding();
    std::uninitialized_fill(begin(), end(), value);
  }

  iterator begin() {
    return data();
  }
  iterator end() {
    return data() + size_;
  }

  const_iterator begin() const {
    return data();
  }
  const_iterator end() const {
    return data() + size_;
  }

  Value& operator[](Size index) {
    return data()[index];
  }
  const Value& operator[](Size index) const {
    return data()[index];
  }

  Value* data() {
    return values_.data() + 1;
  }
  const Value* data() const {
    return values_.data() + 1;
  }

  std::span<Value> span() {
    return std::span{data(), size()};
  }
  std::span<const Value> span() const {
    return std::span{data(), size()};
  }

  [[nodiscard]] Size size() const {
    return size_;
  }

private:
  void init_padding() {
    std::uninitialized_value_construct(values_.begin(), begin());
    std::uninitialized_value_construct(end(), values_.end());
  }

  Data values_;
  Size size_;
};
} // namespace lineal

#endif // INCLUDE_LINEAL_TENSOR_DYNAMIC_STENCIL_STORAGE_CELL_STORAGE_HPP
