// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_PARALLEL_INDEX_INDEX_HPP
#define INCLUDE_LINEAL_PARALLEL_INDEX_INDEX_HPP

#include <cassert>
#include <concepts>
#include <type_traits>

#include "thesauros/macropolis.hpp"
#include "thesauros/ranges.hpp"
#include "thesauros/types.hpp"

#include "lineal/parallel/def/index.hpp"

namespace lineal {
template<std::unsigned_integral TSize, OptIndexTag TIdxTag>
struct OptDistributedIndexTrait;
template<std::unsigned_integral TSize>
struct OptDistributedIndexTrait<TSize, void> {
  using Type = TSize;
};
template<std::unsigned_integral TSize, OptIndexTag TIdxTag>
using OptDistributedIndex = OptDistributedIndexTrait<TSize, TIdxTag>::Type;

template<typename T>
struct IndexRange;
template<std::unsigned_integral TSize>
struct IndexRange<TSize> {
  using Size = TSize;
  using Index = Size;

  explicit constexpr IndexRange(Index begin, Index end) : begin_(begin), end_(end) {}

  [[nodiscard]] constexpr bool contains(Index idx) const {
    return begin_ <= idx && idx < end_;
  }

  [[nodiscard]] constexpr Index begin_index() const {
    return begin_;
  }
  [[nodiscard]] constexpr Index end_index() const {
    return end_;
  }

  [[nodiscard]] constexpr Size size() const {
    return end_ - begin_;
  }

private:
  Index begin_;
  Index end_;
};

template<typename TSize, typename TIdx>
THES_ALWAYS_INLINE inline constexpr auto to_index_range(thes::IotaRange<TSize> r,
                                                        thes::TypeTag<TIdx> /*tag*/) {
  return IndexRange<TIdx>{TIdx{r.begin_value()}, TIdx{r.end_value()}};
}

template<typename T>
struct IsDistributedIndexTrait : public std::false_type {};
template<typename T>
concept AnyDistributedIndex = IsDistributedIndexTrait<T>::value;
template<typename TIdx, typename TLocalSize, typename TGlobalSize>
concept TypedDistributedIndex =
  AnyDistributedIndex<TIdx> && ((std::same_as<typename TIdx::Tag, GlobalIndexTag> &&
                                 std::same_as<typename TIdx::Size, TGlobalSize>) ||
                                (std::same_as<typename TIdx::Tag, LocalIndexTag> &&
                                 std::same_as<typename TIdx::Size, TLocalSize>) ||
                                (std::same_as<typename TIdx::Tag, OwnIndexTag> &&
                                 std::same_as<typename TIdx::Size, TLocalSize>));
template<typename TIdx, bool tIsShared, typename TLocalSize, typename TGlobalSize>
concept TypedIndex = (!tIsShared || std::same_as<TIdx, TLocalSize>) &&
                     (tIsShared || TypedDistributedIndex<TIdx, TGlobalSize, TLocalSize>);
template<typename T>
concept AnyIndex = AnyDistributedIndex<T> || std::unsigned_integral<T>;
template<typename TIdx, typename TLocalSize, typename TGlobalSize>
concept AnyTypedIndex =
  std::same_as<TIdx, TLocalSize> || TypedDistributedIndex<TIdx, TLocalSize, TGlobalSize>;

template<bool tIsShared, typename TSize>
using OptGlobalIndex = TSize;
template<bool tIsShared, typename TSize>
using OptLocalIndex = TSize;
template<bool tIsShared, typename TSize>
using OptOwnIndex = TSize;
template<bool tIsShared, typename TSize, AnyIndexTag TTag>
using OptIndex = TSize;

template<AnyIndex TIdx>
struct IndexSizeTrait;
template<std::unsigned_integral TSize>
struct IndexSizeTrait<TSize> {
  using Size = TSize;
};
template<AnyDistributedIndex TIdx>
struct IndexSizeTrait<TIdx> {
  using Size = TIdx::Size;
};
template<AnyIndex TIdx>
using IndexSize = IndexSizeTrait<TIdx>::Size;

THES_ALWAYS_INLINE inline constexpr auto index_convert(auto index, AnyIndexTag auto tag,
                                                       const auto& dist_info)
requires(!AnyDistributedIndex<decltype(index)> || requires { index.convert_to(tag, dist_info); })
{
  if constexpr (AnyDistributedIndex<decltype(index)>) {
    return index.convert_to(tag, dist_info);
  } else {
    return index;
  }
}
template<AnyIndex TIdx>
THES_ALWAYS_INLINE inline constexpr auto index_convert(auto index, const auto& dist_info,
                                                       thes::TypeTag<TIdx> /*tag*/ = {}) {
  if constexpr (AnyDistributedIndex<decltype(index)>) {
    return index.convert_to(TIdx::tag, dist_info);
  } else {
    return index;
  }
}

THES_ALWAYS_INLINE inline constexpr auto index_value(auto index) {
  if constexpr (AnyDistributedIndex<decltype(index)>) {
    return index.index();
  } else {
    return index;
  }
}
THES_ALWAYS_INLINE inline constexpr auto index_value(auto index, AnyIndexTag auto tag,
                                                     const auto& dist_info)
requires requires { index_convert(index, tag, dist_info); }
{
  return index_value(index_convert(index, tag, dist_info));
}

template<AnyDistributedIndex TIdx>
THES_ALWAYS_INLINE inline constexpr auto index_range(TIdx begin, TIdx end) {
  assert(&begin.distributed_info() == &end.distributed_info());
  return thes::transform_range(
    [&dist_info = begin.distributed_info()](auto v) { return TIdx{dist_info, v}; },
    thes::range(begin.index(), end.index()));
}
} // namespace lineal

#endif // INCLUDE_LINEAL_PARALLEL_INDEX_INDEX_HPP
