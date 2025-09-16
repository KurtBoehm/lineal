// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_AGGREGATE_MAP_MULTITHREAD_HPP
#define INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_AGGREGATE_MAP_MULTITHREAD_HPP

#include <cassert>
#include <cstddef>
#include <latch>
#include <memory>
#include <optional>
#include <utility>

#include "thesauros/containers.hpp"
#include "thesauros/io.hpp"
#include "thesauros/iterator.hpp"
#include "thesauros/math.hpp"
#include "thesauros/types.hpp"
#include "thesauros/utility.hpp"

#include "lineal/linear-solver/multigrid/types.hpp"
#include "lineal/parallel.hpp"
#include "lineal/vectorization.hpp"

namespace lineal::amg {
template<bool tIsShared, typename TSizeByte, typename TByteAlloc>
struct BaseMultithreadAggregateMap {
  using SizeByte = TSizeByte;
  using IndexTag = OptOwnIndexTag<tIsShared>;
  using Size = SizeByte::Unsigned;
  using Index = OptDistributedIndex<Size, IndexTag>;
  using VertexSize = Size;
  using VertexIdx = Index;
  using Aggregate = amg::Aggregate<SizeByte, IndexTag>;
  template<typename T>
  using Alloc = std::allocator_traits<TByteAlloc>::template rebind_alloc<T>;

  using Offsets = thes::FixedArray<Size, thes::DefaultInit, Alloc<Size>>;
  using VertexToAggregateMap =
    thes::MultiByteIntegers<SizeByte, grex::register_bytes.back(), TByteAlloc>;
  using AggregateIndexIter = VertexToAggregateMap::const_iterator;
  using AggregateToVertexMap = thes::NestedDynamicArray<Size, Size, Alloc<Size>>;
  using AggregateToVertexBuilder = AggregateToVertexMap::NestedBuilder;

  static constexpr bool is_shared = tIsShared;

  struct IterProvider {
    using State = AggregateIndexIter;
    struct IterTypes : public thes::iter_provider::ValueTypes<Aggregate, std::ptrdiff_t> {
      using IterState = State;
    };
    using Ref = IterTypes::IterRef;

    static Ref deref(const auto& self) {
      return Aggregate{Index{*self.iter_}};
    }
    template<typename TSelf>
    static thes::TransferConst<TSelf, State>& state(TSelf& self) {
      return self.iter_;
    }
  };

  struct const_iterator
      : public thes::IteratorFacade<const_iterator, thes::iter_provider::Map<IterProvider>> {
    friend struct IterProvider;

    explicit const_iterator(AggregateIndexIter iter) : iter_(iter) {}

    template<typename TVecSize>
    auto load(TVecSize vec_size) const {
      return grex::load_multibyte(iter_, vec_size);
    }

  private:
    AggregateIndexIter iter_;
  };

  struct Builder {
    template<typename TVertices>
    struct ThreadInstance {
      using Size = BaseMultithreadAggregateMap::Size;
      using Index = BaseMultithreadAggregateMap::Index;
      using VertexSize = BaseMultithreadAggregateMap::VertexSize;
      using VertexIdx = BaseMultithreadAggregateMap::VertexIdx;
      using Aggregate = BaseMultithreadAggregateMap::Aggregate;

      using ThreadAggregateToVertexMap = thes::ChunkedDynamicArray<VertexSize>;

      ThreadInstance(std::size_t thread_index, TVertices&& vertices, Offsets& aggregate_nums,
                     Offsets& aggregated_nums, std::latch& barrier_allocate,
                     std::latch& barrier_merge, VertexToAggregateMap& fine_to_coarse,
                     AggregateToVertexBuilder& coarse_to_fine_builder, Size max_aggregate_size)
          : thread_index_(thread_index), vertices_(std::forward<TVertices>(vertices)),
            aggregate_nums_(aggregate_nums), aggregated_nums_(aggregated_nums),
            barrier_merge_(barrier_merge), barrier_allocate_(barrier_allocate),
            fine_to_coarse_(fine_to_coarse), coarse_to_fine_builder_(coarse_to_fine_builder),
            thread_coarse_to_fine_(max_aggregate_size) {}

      ThreadInstance(ThreadInstance&&) = delete;
      ThreadInstance(const ThreadInstance&) = delete;
      ThreadInstance& operator=(ThreadInstance&&) = delete;
      ThreadInstance& operator=(const ThreadInstance&) = delete;

      ~ThreadInstance() {
        aggregate_nums_[thread_index_] = *thes::safe_cast<Size>(thread_coarse_to_fine_.block_num());
        aggregated_nums_[thread_index_] =
          *thes::safe_cast<Size>(thread_coarse_to_fine_.value_num());
        barrier_merge_.arrive_and_wait();

        if (thread_index_ == 0) {
          const Size aggregate_num =
            std::reduce(aggregate_nums_.begin(), aggregate_nums_.end(), Size{0});
          const Size aggregated_num =
            std::reduce(aggregated_nums_.begin(), aggregated_nums_.end(), Size{0});
          coarse_to_fine_builder_.initialize(aggregate_num, aggregated_num);
        }
        barrier_allocate_.arrive_and_wait();

        const Size aggregate_offset =
          std::reduce(aggregate_nums_.begin(), aggregate_nums_.begin() + thread_index_, Size{0});
        if (aggregate_offset != 0) {
          for (const Size vertex : vertices_) {
            decltype(auto) entry = fine_to_coarse_[vertex];
            Aggregate agg(Index{entry});
            if (agg.is_aggregate()) {
              entry = entry + aggregate_offset;
            }
          }
        }

        const Size vertex_offset =
          std::reduce(aggregated_nums_.begin(), aggregated_nums_.begin() + thread_index_, Size{0});
        auto thread_creator = coarse_to_fine_builder_.part_builder(aggregate_offset, vertex_offset);
        for (decltype(auto) block : thread_coarse_to_fine_) {
          for (decltype(auto) vertex : block) {
            thread_creator.emplace(vertex);
          }
          thread_creator.advance_group();
        }
      }

      Aggregate aggregate_of(VertexIdx target) const {
        return Aggregate(Index{fine_to_coarse_[index_value(target)]});
      }
      Size aggregate_size(const Aggregate& agg) const {
        return *thes::safe_cast<Size>(thread_coarse_to_fine_[index_value(agg.index())].size());
      }

      void store_aggregate(VertexIdx target, Aggregate agg) {
        assert(aggregate_of(target).is_unaggregated());
        const auto tidx = index_value(target);
        const auto agg_idx = index_value(agg.index());
        fine_to_coarse_[tidx] = agg_idx;

        decltype(auto) aggregate_block = thread_coarse_to_fine_[agg_idx];
        assert(std::find(aggregate_block.begin(), aggregate_block.end(), tidx) ==
               aggregate_block.end());
        aggregate_block.push_back(tidx);
      }
      void store_isolated(VertexIdx target) {
        assert(aggregate_of(target).is_unaggregated());
        fine_to_coarse_[index_value(target)] = Aggregate::isolated_index;
      }

      void move_to(VertexIdx target, Aggregate agg) {
        const auto tidx = index_value(target);

        decltype(auto) old_entry = fine_to_coarse_[tidx];
        Aggregate old_aggregate(Index{old_entry});
        assert(old_aggregate.is_aggregate());
        assert(old_aggregate.index() != agg.index());

        decltype(auto) old_aggregate_block = thread_coarse_to_fine_[old_entry];
        assert(old_aggregate_block.size() == 1);
        old_aggregate_block.erase(tidx);
        assert(old_aggregate_block.size() == 0);

        old_entry = index_value(agg.index());

        decltype(auto) aggregate_block = thread_coarse_to_fine_[index_value(agg.index())];
        assert(std::find(aggregate_block.begin(), aggregate_block.end(), tidx) ==
               aggregate_block.end());
        aggregate_block.push_back(tidx);
      }
      void move_to_isolated(VertexIdx target) {
        const auto tidx = index_value(target);

        decltype(auto) old_entry = fine_to_coarse_[tidx];
        Aggregate old_aggregate(Index{old_entry});
        assert(old_aggregate.is_aggregate());

        decltype(auto) old_aggregate_block = thread_coarse_to_fine_[old_entry];
        assert(old_aggregate_block.size() == 1);
        old_aggregate_block.erase(tidx);
        assert(old_aggregate_block.size() == 0);

        old_entry = Aggregate::isolated_index;
      }

      void push_aggregate() {
        thread_coarse_to_fine_.push_block();
      }
      void pop_aggregate() {
        thread_coarse_to_fine_.pop_block();
      }

      void add_aggregates(Size agg_num) {
        if (agg_num > 0) {
          thread_coarse_to_fine_.add_blocks(agg_num);
        }
      }

    private:
      void move_to_impl(VertexSize target, Aggregate aggregate) {
        decltype(auto) old_entry = fine_to_coarse_[target];
        Aggregate old_aggregate(Index{old_entry});
        assert(old_aggregate.is_aggregate());
        assert(old_aggregate.index() != aggregate.index());

        decltype(auto) old_aggregate_block = thread_coarse_to_fine_[old_entry];
        assert(old_aggregate_block.size() == 1);
        old_aggregate_block.erase(target);
        assert(old_aggregate_block.size() == 0);

        old_entry = index_value(aggregate.index());
      }

      const std::size_t thread_index_;
      TVertices vertices_;
      Offsets& aggregate_nums_;
      Offsets& aggregated_nums_;
      std::latch& barrier_merge_;
      std::latch& barrier_allocate_;
      VertexToAggregateMap& fine_to_coarse_;
      AggregateToVertexBuilder& coarse_to_fine_builder_;

      ThreadAggregateToVertexMap thread_coarse_to_fine_;
    };

    struct ExecInstance {
      template<typename TVertices>
      using ThreadInstance = ThreadInstance<TVertices>;

      ExecInstance(std::size_t thread_num, VertexToAggregateMap& fine_to_coarse,
                   AggregateToVertexBuilder& coarse_to_fine_builder)
          : fine_to_coarse_(fine_to_coarse), coarse_to_fine_builder_(coarse_to_fine_builder),
            aggregate_nums_(thread_num, 0), aggregated_nums_(thread_num, 0),
            barrier_allocate_(*thes::safe_cast<std::ptrdiff_t>(thread_num)),
            barrier_merge_(*thes::safe_cast<std::ptrdiff_t>(thread_num)) {}

      template<typename TVertices>
      ThreadInstance<TVertices> thread_instance(std::size_t thread_index, TVertices&& vertices,
                                                Size max_aggregate_size) {
        return {
          thread_index,       std::forward<TVertices>(vertices),
          aggregate_nums_,    aggregated_nums_,
          barrier_allocate_,  barrier_merge_,
          fine_to_coarse_,    coarse_to_fine_builder_,
          max_aggregate_size,
        };
      }

    private:
      VertexToAggregateMap& fine_to_coarse_;
      AggregateToVertexBuilder& coarse_to_fine_builder_;

      Offsets aggregate_nums_;
      Offsets aggregated_nums_;
      std::latch barrier_allocate_;
      std::latch barrier_merge_;
    };

    explicit Builder(VertexIdx vertex_index_end) : fine_to_coarse_(index_value(vertex_index_end)) {}

    void initialize(const auto& expo, std::optional<std::size_t> thread_num) {
      expo.execute_segmented(
        fine_to_coarse_.size(),
        [&](auto /*tidx*/, auto begin, auto end) { fine_to_coarse_.set_all(begin, end); },
        thread_num);
    }

    ExecInstance exec_instance(std::size_t thread_num) {
      return ExecInstance(thread_num, fine_to_coarse_, coarse_to_fine_builder_);
    }

    BaseMultithreadAggregateMap build() {
      return BaseMultithreadAggregateMap(std::move(fine_to_coarse_),
                                         coarse_to_fine_builder_.build());
    }

  private:
    VertexToAggregateMap fine_to_coarse_;
    AggregateToVertexBuilder coarse_to_fine_builder_{};
  };

  static BaseMultithreadAggregateMap from_file(thes::FileReader& reader) {
    auto fine_to_coarse = VertexToAggregateMap::from_file(reader);
    auto coarse_to_fine = AggregateToVertexMap::from_file(reader);
    return BaseMultithreadAggregateMap(std::move(fine_to_coarse), std::move(coarse_to_fine));
  }

  BaseMultithreadAggregateMap(const BaseMultithreadAggregateMap&) = delete;
  BaseMultithreadAggregateMap& operator=(const BaseMultithreadAggregateMap&) = delete;
  BaseMultithreadAggregateMap(BaseMultithreadAggregateMap&&) noexcept = default;
  BaseMultithreadAggregateMap& operator=(BaseMultithreadAggregateMap&&) noexcept = default;

  ~BaseMultithreadAggregateMap() = default;

  [[nodiscard]] Size fine_row_num() const {
    return *thes::safe_cast<Size>(fine_to_coarse_.size());
  }
  [[nodiscard]] Size coarse_row_num() const {
    return *thes::safe_cast<Size>(coarse_to_fine_.group_num());
  }

  Aggregate operator[](VertexSize vertex) const {
    assert(vertex < fine_to_coarse_.size());
    return Aggregate(Index{fine_to_coarse_[vertex]});
  }

  auto load(VertexSize vertex, grex::AnyTag auto tag) const {
    assert(vertex < fine_to_coarse_.size());
    const auto vec = grex::load_multibyte(fine_to_coarse_.begin() + vertex, tag);
    return AggregateVector<SizeByte, IndexTag, tag.size>(vec);
  }

  const_iterator begin() const {
    return const_iterator(fine_to_coarse_.begin());
  }
  const_iterator end() const {
    return const_iterator(fine_to_coarse_.end());
  }

  auto coarse_row_to_fine_rows(const Aggregate& agg) const {
    if constexpr (is_shared) {
      return coarse_to_fine_[agg.index()];
    } else {
      return thes::transform_range([](auto idx) { return VertexIdx{idx}; },
                                   coarse_to_fine_[index_value(agg.index())]);
    }
  }

  void to_file(thes::FileWriter& writer) {
    fine_to_coarse_.to_file(writer);
    coarse_to_fine_.to_file(writer);
  }

private:
  BaseMultithreadAggregateMap(VertexToAggregateMap&& fine_to_coarse,
                              AggregateToVertexMap&& coarse_to_fine)
      : fine_to_coarse_(std::forward<VertexToAggregateMap>(fine_to_coarse)),
        coarse_to_fine_(std::forward<AggregateToVertexMap>(coarse_to_fine)) {}

  VertexToAggregateMap fine_to_coarse_;
  AggregateToVertexMap coarse_to_fine_;
};

template<typename TSizeByte, typename TByteAlloc>
using MultithreadAggregateMap = BaseMultithreadAggregateMap<true, TSizeByte, TByteAlloc>;
} // namespace lineal::amg

#endif // INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_AGGREGATE_MAP_MULTITHREAD_HPP
