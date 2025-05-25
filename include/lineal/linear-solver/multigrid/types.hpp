// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_TYPES_HPP
#define INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_TYPES_HPP

#include <cassert>
#include <compare>
#include <concepts>
#include <cstddef>
#include <limits>

#include "thesauros/macropolis.hpp"
#include "thesauros/types.hpp"
#include "thesauros/utility.hpp"

#include "lineal/parallel/def/index.hpp"
#include "lineal/parallel/index/index.hpp"
#include "lineal/vectorization.hpp"

namespace lineal::amg {
template<typename TSize>
struct ClosedRange {
  using Size = TSize;
  THES_DEFINE_TYPE(SNAKE_CASE(ClosedRange), NO_CONSTRUCTOR,
                   MEMBERS((KEEP(min), Size), (KEEP(max), Size)))
};

template<typename TIdx>
struct Vertex {
  using Index = TIdx;
  using Size = IndexSize<Index>;

  explicit Vertex(Index idx) : idx_(idx) {}

  [[nodiscard]] Index index() const {
    return idx_;
  }

  friend bool operator==(const Vertex& v1, const Vertex& v2) = default;
  friend std::strong_ordering operator<=>(const Vertex& v1, const Vertex& v2) = default;

private:
  Index idx_;
};

template<typename TVtxIdx, std::unsigned_integral TEdgeSize>
struct Edge {
  using VertexIdx = TVtxIdx;
  using Vertex = amg::Vertex<VertexIdx>;
  using EdgeSize = TEdgeSize;
  using OptEdgeSize = thes::ValueOptional<TEdgeSize, std::numeric_limits<EdgeSize>::max()>;

  Edge(OptEdgeSize idx, VertexIdx tail, VertexIdx head) : idx_(idx), tail_(tail), head_(head) {};

  [[nodiscard]] OptEdgeSize index() const {
    return idx_;
  }
  [[nodiscard]] VertexIdx tail_index() const {
    return tail_;
  }
  [[nodiscard]] VertexIdx head_index() const {
    return head_;
  }

  [[nodiscard]] Vertex tail() const {
    return Vertex{tail_};
  }
  [[nodiscard]] Vertex head() const {
    return Vertex{head_};
  }

private:
  OptEdgeSize idx_;
  VertexIdx tail_;
  VertexIdx head_;
};

template<typename TSizeByte, OptIndexTag TIdxTag = void>
struct Aggregate {
  using Size = TSizeByte::Unsigned;
  using Index = OptDistributedIndex<Size, TIdxTag>;

  static constexpr Size unaggregated_index = TSizeByte::max;
  static constexpr Size isolated_index = TSizeByte::max - 1;

  static constexpr Aggregate unaggregated() {
    return Aggregate{unaggregated_index};
  };
  static constexpr Aggregate isolated() {
    return Aggregate{isolated_index};
  };

  constexpr Aggregate() = default;
  explicit constexpr Aggregate(Index index) : index_(index_value(index)) {}

  constexpr Index index() const {
    assert(is_aggregate());
    return Index{index_};
  }

  constexpr Index operator*() const {
    return Index{index_};
  }

  [[nodiscard]] constexpr bool is_aggregate() const {
    return index_ < isolated_index;
  }
  [[nodiscard]] constexpr bool is_aggregated() const {
    return index_ != unaggregated_index;
  }
  [[nodiscard]] constexpr bool is_unaggregated() const {
    return index_ == unaggregated_index;
  }
  [[nodiscard]] constexpr bool is_isolated() const {
    return index_ == isolated_index;
  }

  // For an optional-like interface — use with caution!
  [[nodiscard]] constexpr bool has_value() const {
    return is_aggregated();
  }

  template<typename TOutSizeByte = TSizeByte, typename TOutIdxTag = thes::VoidStorage<TIdxTag>>
  auto transform(auto op, thes::TypeTag<TOutSizeByte> /*size_tag*/ = {},
                 TOutIdxTag /*index_tag*/ = {}) const
  requires(std::same_as<
           decltype(op(std::declval<Index>())),
           OptDistributedIndex<typename TOutSizeByte::Unsigned, thes::UnVoidStorage<TOutIdxTag>>>)
  {
    using OutSize = typename TOutSizeByte::Unsigned;
    using OutIdx = OptDistributedIndex<OutSize, thes::UnVoidStorage<TOutIdxTag>>;

    auto out_limits = [](Size idx) {
      if constexpr (std::same_as<TOutSizeByte, TSizeByte>) {
        return idx;
      } else if constexpr (TOutSizeByte::byte_num > TSizeByte::byte_num) {
        constexpr OutSize mask = TOutSizeByte::max & ~OutSize{TSizeByte::max};
        return OutSize{idx} | mask;
      } else if constexpr (TOutSizeByte::byte_num < TSizeByte::byte_num) {
        return OutSize{idx} & TOutSizeByte::max;
      }
    };
    static_assert(out_limits(TSizeByte::max) == TOutSizeByte::max);
    static_assert(out_limits(TSizeByte::max - 1) == TOutSizeByte::max - 1);

    using Ret = Aggregate<TOutSizeByte, thes::UnVoidStorage<TOutIdxTag>>;
    return is_aggregate() ? Ret{op(index())} : Ret{OutIdx{out_limits(index_)}};
  }
  template<typename TOutSizeByte = TSizeByte, typename TOutIdxTag = thes::VoidStorage<TIdxTag>>
  auto offset(TOutSizeByte::Unsigned offset, thes::TypeTag<TOutSizeByte> size_tag = {},
              TOutIdxTag index_tag = {}) const {
    using OutSize = typename TOutSizeByte::Unsigned;
    using OutIdx = OptDistributedIndex<OutSize, thes::UnVoidStorage<TOutIdxTag>>;
    return transform([offset](Index agg) { return OutIdx{index_value(agg) + offset}; }, size_tag,
                     index_tag);
  }

  template<typename TOutSizeByte = TSizeByte, typename TOutIdxTag = thes::VoidStorage<TIdxTag>>
  auto cast(thes::TypeTag<TOutSizeByte> size_tag = {}, TOutIdxTag index_tag = {}) const {
    using OutSize = typename TOutSizeByte::Unsigned;
    using OutIdx = OptDistributedIndex<OutSize, thes::UnVoidStorage<TOutIdxTag>>;
    assert(!is_aggregate() || index_ < TOutSizeByte::max);
    return transform([](Index agg) { return OutIdx{*thes::safe_cast<OutSize>(index_value(agg))}; },
                     size_tag, index_tag);
  }

  friend bool operator==(const Aggregate& agg1, const Aggregate& agg2) {
    return agg1.index_ == agg2.index_;
  }
  friend std::strong_ordering operator<=>(const Aggregate& agg1, const Aggregate& agg2) {
    return agg1.index_ <=> agg2.index_;
  }

private:
  Size index_{unaggregated_index};
};

template<typename TSizeByte, OptIndexTag TIdxTag, std::size_t tSize>
struct AggregateVector {
  using Size = TSizeByte::Unsigned;
  using SizeVector = grex::Vector<Size, tSize>;

  explicit constexpr AggregateVector(SizeVector ids) : ids_(ids) {}

  [[nodiscard]] SizeVector operator*() const {
    return ids_;
  }

  [[nodiscard]] grex::Mask<Size, tSize> is_aggregate() const {
    return ids_ < SizeVector(Aggregate<TSizeByte, TIdxTag>::isolated_index);
  }

private:
  SizeVector ids_;
};
} // namespace lineal::amg

#endif // INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_TYPES_HPP
