// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_TENSOR_DYNAMIC_DENSE_VECTOR_SHARED_HPP
#define INCLUDE_LINEAL_TENSOR_DYNAMIC_DENSE_VECTOR_SHARED_HPP

#include <cassert>
#include <concepts>
#include <cstddef>
#include <memory>
#include <span>
#include <type_traits>
#include <utility>

#include "thesauros/algorithms.hpp"
#include "thesauros/containers.hpp"
#include "thesauros/iterator.hpp"
#include "thesauros/memory.hpp"
#include "thesauros/ranges.hpp"
#include "thesauros/types.hpp"
#include "thesauros/utility.hpp"

#include "lineal/base.hpp"
#include "lineal/parallel.hpp"
#include "lineal/vectorization.hpp"

#ifndef NDEBUG
#include <limits>
#endif

namespace lineal {
template<typename TValue, typename TSize, std::size_t tPadding, typename TAllocator,
         OptDistributedInfo TDistInfo = void>
struct DenseVectorBase : public SharedVectorBase {
  using Value = TValue;
  using Size = TSize;
  using Allocator = TAllocator;
  static constexpr std::size_t padding = tPadding;

  using RawDistributedInfo = TDistInfo;
  using DistributedInfo = std::decay_t<TDistInfo>;
  using DistributedInfoStorage = thes::VoidStorage<TDistInfo>;
  static constexpr bool is_shared = std::is_void_v<TDistInfo>;

  using GlobalSize = GlobalSizeOf<DistributedInfo, Size>;
  using Index = OptLocalIndex<is_shared, Size>;
  using ExtIndex = Index;

  using Data = thes::array::TypedChunk<Value, Size, Allocator>;

private:
  template<bool tIsConst>
  struct IterProvider {
    using State = thes::ConditionalConst<tIsConst, Value>*;
    struct IterTypes
        : public thes::iter_provider::DefaultTypes<thes::ConditionalConst<tIsConst, Value>,
                                                   std::ptrdiff_t> {
      using IterState = State;
    };
    using Ref = IterTypes::IterRef;

    static Ref deref(const auto& self) {
      return *self.ptr_;
    }
    template<typename TSelf>
    static thes::TransferConst<TSelf, State>& state(TSelf& self) {
      return self.ptr_;
    }
  };

public:
  struct Iterator;

  struct ConstIterator
      : public thes::IteratorFacade<ConstIterator, thes::iter_provider::Map<IterProvider<true>>> {
    friend struct IterProvider<true>;
    using DistributedInfoStoragePtr = thes::VoidStorageConstPtr<DistributedInfo>;

    ConstIterator() = default;
    explicit ConstIterator(const Value* ptr, DistributedInfoStoragePtr dist_info)
        : ptr_(ptr), dist_info_(dist_info) {}
    explicit ConstIterator(const Iterator& other)
        : ptr_{other.ptr_}, dist_info_{other.dist_info_} {}

    auto compute(TensorTag<Value> auto tag) const {
      return load_ptr(ptr_, tag);
    }

  private:
    const Value* ptr_{};
    [[no_unique_address]] DistributedInfoStoragePtr dist_info_{};
  };

  struct Iterator
      : public thes::IteratorFacade<Iterator, thes::iter_provider::Map<IterProvider<false>>> {
    friend struct IterProvider<false>;
    friend struct ConstIterator;
    using DistributedInfoStoragePtr = thes::VoidStorageConstPtr<DistributedInfo>;

    Iterator() = default;
    explicit Iterator(Value* ptr, DistributedInfoStoragePtr dist_info)
        : ptr_(ptr), dist_info_(dist_info) {}

    auto compute(TensorTag<Value> auto tag) const {
      return load_ptr(ptr_, tag);
    }

    void store(auto vector, TensorTag<Value> auto tag) {
      compat::store(ptr_, vector, tag);
    }

  private:
    Value* ptr_{};
    [[no_unique_address]] DistributedInfoStoragePtr dist_info_{};
  };

  using const_iterator = ConstIterator;
  using iterator = Iterator;

  DenseVectorBase(const Size size, DistributedInfoStorage&& dist_info)
      : values_(size + 2 * padding), size_(size),
        dist_info_(std::forward<DistributedInfoStorage>(dist_info)) {
    auto* data_begin = data();
    auto* data_end = data_begin + size;

    std::uninitialized_value_construct(values_.begin(), data_begin);
    std::uninitialized_value_construct(data_end, values_.end());
#ifndef NDEBUG
    std::uninitialized_fill(data_begin, data_end, std::numeric_limits<Value>::signaling_NaN());
#endif
  }

  DenseVectorBase(const Size size, DistributedInfoStorage&& dist_info, const Env auto& env)
      : values_(size + 2 * padding), size_(size),
        dist_info_(std::forward<DistributedInfoStorage>(dist_info)) {
    auto* data_begin = data();
    auto* data_end = data_begin + size;

    std::uninitialized_value_construct(values_.begin(), data_begin);
    std::uninitialized_value_construct(data_end, values_.end());

    env.execution_policy().execute_segmented(
      size, [&](std::size_t /*tidx*/, auto first, auto last) {
        std::uninitialized_value_construct(data_begin + first, data_begin + last);
      });
  }

  DenseVectorBase()
  requires(is_shared)
  = default;

  DenseVectorBase(const Size size, const Env auto& env)
  requires(is_shared)
      : DenseVectorBase{size, thes::Empty{}, env} {}

  DenseVectorBase(const Size size, Value value, DistributedInfoStorage&& dist_info,
                  const Env auto& env)
      : values_(size + 2 * padding), size_(size),
        dist_info_(std::forward<DistributedInfoStorage>(dist_info)) {
    auto* data_begin = data();
    auto* data_end = data_begin + size;

    std::uninitialized_value_construct(values_.begin(), data_begin);
    std::uninitialized_value_construct(data_end, values_.end());

    env.execution_policy().execute_segmented(
      size, [&](std::size_t /*tidx*/, auto first, auto last) {
        std::uninitialized_fill(data_begin + first, data_begin + last, value);
      });
  }

  DenseVectorBase(const Size size, Value value, const Env auto& env)
  requires(is_shared)
      : DenseVectorBase(size, value, thes::Empty{}, env) {}

  DenseVectorBase(Data&& data, Size size, DistributedInfoStorage&& dist_info)
      : values_(std::forward<Data>(data)), size_(size),
        dist_info_(std::forward<DistributedInfoStorage>(dist_info)) {
    assert(values_.size() >= size + 2 * padding);
  }

  DenseVectorBase(DenseVectorBase&& other) noexcept
      : values_(std::move(other.values_)), size_(other.size_),
        dist_info_(std::move(other.dist_info_)) {
    other.size_ = 0;
  }
  DenseVectorBase(const DenseVectorBase& other)
      : values_(other.values_.size()), size_(other.size_), dist_info_(other.dist_info_) {
    std::uninitialized_copy(other.values_.begin(), other.values_.end(), values_.begin());
  }

  DenseVectorBase& operator=(DenseVectorBase&& other) noexcept {
    destroy();
    values_.move_to_destroyed(std::move(other.values_));
    size_ = other.size_;
    dist_info_ = std::move(other.dist_info_);
    other.size_ = 0;
    return *this;
  }
  DenseVectorBase& operator=(const DenseVectorBase& other) {
    if (this != &other) {
      destroy();
      values_.reallocate_to_destroyed(other.values_);
      std::uninitialized_copy(other.begin(), other.end(), begin());
      size_ = other.size_;
      dist_info_ = other.dist_info_;
    }
    return *this;
  }

  ~DenseVectorBase() {
    destroy();
  }

  friend void swap(DenseVectorBase& v1, DenseVectorBase& v2) {
    using std::swap;
    swap(v1.values_, v2.values_);
    swap(v1.size_, v2.size_);
    thes::swap_or_equal<DistributedInfoStorage>(v1.dist_info_, v2.dist_info_);
  }

  [[nodiscard]] Value* data() {
    return values_.begin() + padding;
  }
  [[nodiscard]] const Value* data() const {
    return values_.begin() + padding;
  }

  [[nodiscard]] std::span<Value> span() {
    return {data(), size()};
  }
  [[nodiscard]] std::span<const Value> span() const {
    return {data(), size()};
  }

  [[nodiscard]] std::span<Value> span(thes::IotaRange<Size> range) {
    return {data() + range.begin_value(), range.size()};
  }
  [[nodiscard]] std::span<const Value> span(thes::IotaRange<Size> range) const {
    return {data() + range.begin_value(), range.size()};
  }

  [[nodiscard]] Iterator begin() {
    return Iterator{data(), thes::void_storage_cptr(dist_info_)};
  }
  [[nodiscard]] ConstIterator begin() const {
    return ConstIterator{data(), thes::void_storage_cptr(dist_info_)};
  }

  [[nodiscard]] Iterator end() {
    return Iterator{data() + size_, thes::void_storage_cptr(dist_info_)};
  }
  [[nodiscard]] ConstIterator end() const {
    return ConstIterator{data() + size_, thes::void_storage_cptr(dist_info_)};
  }
  Iterator iter_at(TypedIndex<is_shared, Size, GlobalSize> auto i) {
    const auto idx = index_value(i, local_index_tag, dist_info_);
    return Iterator{data() + idx, thes::void_storage_cptr(dist_info_)};
  }
  ConstIterator iter_at(TypedIndex<is_shared, Size, GlobalSize> auto i) const {
    const auto idx = index_value(i, local_index_tag, dist_info_);
    return ConstIterator{data() + idx, thes::void_storage_cptr(dist_info_)};
  }

  [[nodiscard]] Value& operator[](TypedIndex<is_shared, Size, GlobalSize> auto i) {
    const auto index = index_value(i, local_index_tag, dist_info_);
    assert(index < size_);
    return data()[index];
  }
  [[nodiscard]] Value operator[](TypedIndex<is_shared, Size, GlobalSize> auto i) const {
    const auto index = index_value(i, local_index_tag, dist_info_);
    assert(index < size_);
    return data()[index];
  }

  decltype(auto) compute(TypedIndex<is_shared, Size, GlobalSize> auto i,
                         TensorTag<Value> auto tag) const {
    const auto index = index_value(i, local_index_tag, dist_info_);
    assert(index < size_);
    assert(grex::is_load_valid(size_ - index, tag));
    return load_ptr(data() + index, tag);
  }
  decltype(auto) compute(TypedIndex<is_shared, Size, GlobalSize> auto i,
                         TensorTag<Value> auto tag) {
    const auto index = index_value(i, local_index_tag, dist_info_);
    assert(index < size_);
    assert(grex::is_load_valid(size_ - index, tag));
    return load_ptr(data() + index, tag);
  }
  // TODO
  decltype(auto) compute(thes::AnyIndexPosition auto i, TensorTag<Value> auto tag) const {
    return compute(i.index, tag);
  }
  decltype(auto) compute(thes::AnyIndexPosition auto i, TensorTag<Value> auto tag) {
    return compute(i.index, tag);
  }
  decltype(auto) lookup(auto idxs, TensorTag<Value> auto tag) const {
    return grex::gather(span(), idxs, tag);
  }

  void store(TypedIndex<is_shared, Size, GlobalSize> auto i, auto v, TensorTag<Value> auto tag) {
    const auto index = index_value(i, local_index_tag, dist_info_);
    grex::store(data() + index, v, tag);
  }
  // TODO
  void store(thes::AnyIndexPosition auto i, auto v, TensorTag<Value> auto tag) {
    store(i.index, v, tag);
  }

  [[nodiscard]] Size size() const {
    return size_;
  }

  const DistributedInfoStorage& distributed_info_storage() const {
    return dist_info_;
  }
  const DistributedInfoStorage& distributed_info() const
  requires(!is_shared)
  {
    return distributed_info_storage();
  }

private:
  template<typename TSrc>
  requires(std::same_as<std::remove_const_t<TSrc>, Value>)
  static decltype(auto) load_ptr(TSrc* src, TensorTag<Value> auto tag) {
    if constexpr (tag.size <= padding) {
      return compat::load_extended(src, tag);
    } else {
      return compat::load(src, tag);
    }
  }

  void destroy() {
    if (size_ != 0) {
      Value* ptr = data();
      std::destroy(ptr, ptr + size());
    }
  }

  Data values_{};
  Size size_{0};

  [[no_unique_address]] DistributedInfoStorage dist_info_{};
};

template<typename TValue, typename TSize, std::size_t tPadding = simd_pad_size<TValue>,
         typename TAllocator = thes::HugePagesAllocator<TValue>>
struct DenseVector : public DenseVectorBase<TValue, TSize, tPadding, TAllocator> {
  using Base = DenseVectorBase<TValue, TSize, tPadding, TAllocator>;
  using Base::Base;

  DenseVector() : Base{} {}
  explicit DenseVector(const TSize size) : Base(size, {}) {}
  explicit DenseVector(const TSize size, TValue value, const Env auto& env)
      : Base(size, value, {}, env) {}
  explicit DenseVector(Base::Data&& data, TSize size)
      : DenseVector(std::forward<Base::Data>(data), size, {}) {}
};
} // namespace lineal

#endif // INCLUDE_LINEAL_TENSOR_DYNAMIC_DENSE_VECTOR_SHARED_HPP
