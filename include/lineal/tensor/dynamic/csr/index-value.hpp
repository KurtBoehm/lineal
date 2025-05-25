// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_TENSOR_DYNAMIC_CSR_INDEX_VALUE_HPP
#define INCLUDE_LINEAL_TENSOR_DYNAMIC_CSR_INDEX_VALUE_HPP

#include <cassert>
#include <cstddef>
#include <iterator>

#include "thesauros/iterator.hpp"
#include "thesauros/types.hpp"

namespace lineal::csr {
template<typename TColIter, typename TValue>
struct IndexValuePair {
  using ColumnIter = TColIter;
  using Column = std::iterator_traits<ColumnIter>::value_type;
  using Value = TValue;

  IndexValuePair(ColumnIter columns_ptr, Value* elements_ptr)
      : column_(*columns_ptr), element_(*elements_ptr), columns_ptr_(columns_ptr),
        elements_ptr_(elements_ptr) {}

  IndexValuePair(const IndexValuePair&) = delete;
  IndexValuePair(IndexValuePair&& other) noexcept = default;
  IndexValuePair& operator=(const IndexValuePair&) = delete;
  IndexValuePair& operator=(IndexValuePair&& other) noexcept {
    column_ = *columns_ptr_ = other.column_;
    element_ = *elements_ptr_ = other.element_;
    return *this;
  }

  ~IndexValuePair() = default;

  friend void swap(const IndexValuePair& ivp1, IndexValuePair&& ivp2) {
    std::iter_swap(ivp1.columns_ptr_, ivp2.columns_ptr_);
    std::iter_swap(ivp1.elements_ptr_, ivp2.elements_ptr_);
  }
  friend void swap(IndexValuePair&& ivp1, IndexValuePair&& ivp2) {
    std::iter_swap(ivp1.columns_ptr_, ivp2.columns_ptr_);
    std::iter_swap(ivp1.elements_ptr_, ivp2.elements_ptr_);
  }

  friend bool operator<(const IndexValuePair& ivp1, const IndexValuePair& ivp2) {
    return ivp1.column_ < ivp2.column_;
  }

private:
  Column column_;
  Value element_;
  ColumnIter columns_ptr_;
  Value* elements_ptr_;
};

template<typename TColIter, typename TValue, typename TState>
struct IndexValueIterProvider {
  using Val = IndexValuePair<TColIter, TValue>;
  using State = TState;

  struct IterTypes : public thes::iter_provider::ValueTypes<Val, std::ptrdiff_t> {
    using IterState = State;
  };

  static Val deref(const auto& self) {
    return IndexValuePair(self.columns_begin_ + self.non_zero_offset_,
                          self.elements_begin_ + self.non_zero_offset_);
  }
  template<typename TSelf>
  static thes::TransferConst<TSelf, State>& state(TSelf& self) {
    return self.non_zero_offset_;
  }

  static void test_if_cmp(const auto& i1, const auto& i2) {
    assert(i1.columns_begin_ == i2.columns_begin_);
    assert(i1.elements_begin_ == i2.elements_begin_);
  }
};

template<typename TColIter, typename TValue, typename TSize>
struct IndexValueIter
    : public thes::IteratorFacade<
        IndexValueIter<TColIter, TValue, TSize>,
        thes::iter_provider::Map<IndexValueIterProvider<TColIter, TValue, TSize>>> {
  friend struct IndexValueIterProvider<TColIter, TValue, TSize>;
  using ColumnIter = TColIter;
  using Mapped = TValue;
  using Size = TSize;

  IndexValueIter(Size non_zero_offset, ColumnIter columns_begin, Mapped* elements_begin)
      : non_zero_offset_(non_zero_offset), columns_begin_(columns_begin),
        elements_begin_(elements_begin) {}

private:
  Size non_zero_offset_;
  ColumnIter columns_begin_;
  Mapped* elements_begin_;
};
} // namespace lineal::csr

#endif // INCLUDE_LINEAL_TENSOR_DYNAMIC_CSR_INDEX_VALUE_HPP
