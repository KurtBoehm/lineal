// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_TENSOR_DYNAMIC_CSR_SHARED_HPP
#define INCLUDE_LINEAL_TENSOR_DYNAMIC_CSR_SHARED_HPP

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>

#include "ankerl/unordered_dense.h"
#include "thesauros/containers.hpp"
#include "thesauros/iterator.hpp"
#include "thesauros/macropolis.hpp"
#include "thesauros/memory.hpp"
#include "thesauros/ranges.hpp"
#include "thesauros/types.hpp"
#include "thesauros/utility.hpp"

#include "lineal/base.hpp"
#include "lineal/parallel.hpp"
#include "lineal/tensor/dynamic/csr/index-value.hpp"
#include "lineal/vectorization.hpp"

namespace lineal {
// When DistributedInfoOf<TDefs> is void, the indices are of type Size.
// When it is not void, row and column indices are of type OwnIndex and LocalIndex, respectively.
template<typename TDefs>
struct CsrMatrixBase : public SharedMatrixBase {
  using Defs = TDefs;
  using Value = Defs::Value;
  using Diff = std::ptrdiff_t;
  using Alloc = Defs::Alloc;
  using ByteAlloc = std::allocator_traits<Alloc>::template rebind_alloc<std::byte>;
  using DistributedInfo = DistributedInfoOf<Defs>;
  using DistributedInfoStorage = thes::VoidStorage<DistributedInfo>;
  using DistributedInfoCref = thes::VoidConstLvalRef<DistributedInfo>;
  using DistributedInfoCrefStorage = thes::VoidStorage<DistributedInfoCref>;
  static constexpr bool is_shared = std::is_void_v<DistributedInfo>;

  using NonZeroSizeByte = Defs::NonZeroSizeByte;
  using NonZeroSize = NonZeroSizeByte::Unsigned;
  using RowOffsets =
    thes::MultiByteIntegers<NonZeroSizeByte, grex::register_bytes.back(), ByteAlloc>;
  using RowOffsetsIter = RowOffsets::iterator;
  using RowOffsetsConstIter = RowOffsets::const_iterator;

  using SizeByte = Defs::SizeByte;
  using Size = SizeByte::Unsigned;
  using Sizes = thes::MultiByteIntegers<SizeByte, grex::register_bytes.back(), ByteAlloc>;
  using GlobalSize = GlobalSizeOf<DistributedInfo, Size>;
  using ColumnIndices = Sizes;
  using ColumnIndexIter = ColumnIndices::iterator;
  using ColumnIndexConstIter = ColumnIndices::const_iterator;

  using RowIdx = OptOwnIndex<is_shared, Size>;
  using ExtRowIdx = OptLocalIndex<is_shared, Size>;
  using ColumnIdx = ExtRowIdx;

  using Entries = thes::DynamicArray<Value, thes::DefaultInit, thes::DoublingGrowth, Alloc>;
  static constexpr std::size_t entry_pad = grex::native_sizes<Value>.back();

private:
  struct ColIterProvider {
    using State = Size;
    struct IterTypes : public thes::iter_provider::ValueTypes<Value, std::ptrdiff_t> {
      using IterState = State;
    };

    static Value deref(const auto& self) {
      return self.val_[self.offset_];
    }
    template<typename TSelf>
    static thes::TransferConst<TSelf, State>& state(TSelf& self) {
      return self.offset_;
    }

    static void test_if_cmp([[maybe_unused]] const auto& i1, [[maybe_unused]] const auto& i2) {
      assert(i1.val_ == i2.val_);
      assert(i1.col_ == i2.col_);
    }
  };

public:
  struct ColumnIterator
      : public thes::IteratorFacade<ColumnIterator, thes::iter_provider::Map<ColIterProvider>> {
    friend struct ColIterProvider;
    using ConstColIter = ColumnIndices::const_iterator;

    ColumnIterator() = default;
    ColumnIterator(const Value* val, ConstColIter col, Size offset)
        : val_{val}, col_{col}, offset_{offset} {}

    [[nodiscard]] auto index() const {
      return ColumnIdx{col_[offset_]};
    }

    const Value* value_it() const {
      return val_ + offset_;
    }
    ConstColIter column_it() const {
      return col_ + offset_;
    }

  private:
    const Value* val_{nullptr};
    ConstColIter col_{};
    Size offset_{0};
  };

  template<bool tExtended>
  struct BaseConstRow {
    using Idx = std::conditional_t<tExtended, ExtRowIdx, RowIdx>;
    using ColumnIndexConstIter = ColumnIndices::const_iterator;

    using const_iterator = ColumnIterator;

    BaseConstRow(const Value* val, ColumnIndexConstIter col, Idx row_idx, Size nnz_size,
                 DistributedInfoCrefStorage dist_info)
        : entry_{val}, col_it_{col}, row_idx_{row_idx}, nnz_size_(nnz_size), dist_info_(dist_info) {
      assert(nnz_size > 0);
    }

    const_iterator begin() const {
      return begin_impl(*this);
    }
    const_iterator end() const {
      return end_impl(*this);
    }

    Value back() const {
      const Size s = size();
      assert(s != 0);
      return entry_[s - 1];
    }
    ColumnIdx back_column() const {
      const Size s = size();
      assert(s != 0);
      return ColumnIdx{col_it_[s - 1]};
    }

    auto column_indices() const {
      return std::ranges::subrange{col_it_, col_it_ + size()};
    }

    const_iterator find(const ColumnIdx i) const {
      return iter_at_index(*this, i);
    }
    [[nodiscard]] bool contains(const ColumnIdx i) const {
      return contains_impl(*this, i);
    }
    [[nodiscard]] Size
    offdiagonal_index_within(const TypedIndex<is_shared, Size, GlobalSize> auto idx) const {
      const auto i = index_convert(idx, local_index_tag, dist_info_);
      const auto it = find(i);

      assert(contains(col_idx()));
      assert(it != end());

      return *thes::safe_cast<Size>(it - begin() - Diff{i > col_idx()});
    }

    Value operator[](const ColumnIdx i) const {
      return at_index(*this, i);
    }

    [[nodiscard]] Idx index() const {
      return row_idx_;
    }
    [[nodiscard]] ColumnIdx ext_index() const {
      return index_convert(row_idx_, local_index_tag, dist_info_);
    }

    [[nodiscard]] Size size() const {
      return nnz_size_;
    }
    [[nodiscard]] Size offdiagonal_num() const {
      assert(contains(col_idx()));
      return size() - 1;
    }

    constexpr decltype(auto) iterate(auto op, AnyValuationTag auto is_valued,
                                     AnyOrderingTag auto /*is_ordered*/) const {
      auto lambda = [&](const_iterator it) THES_ALWAYS_INLINE -> decltype(auto) {
        if constexpr (is_valued) {
          return op(it.index(), *it);
        } else {
          return op(it.index());
        }
      };

      using Ret = decltype(lambda(begin()));
      for (const_iterator j : thes::iter_range(*this)) {
        THES_APPLY_VALUED_RETURN(Ret, lambda(j));
      }
      THES_RETURN_EMPTY_OPTIONAL(Ret);
    }

    constexpr void multicolumn_iterate(auto op, AnyValuationTag auto is_valued,
                                       AnyOrderingTag auto /*is_ordered*/,
                                       grex::FullVectorTag auto vtag) const {
      constexpr Size vsize = vtag.size;
      const Size size = nnz_size_;
      const Size vnum = size / vsize * vsize;

      const Value* vit = entry_;
      ColumnIndexConstIter cit = col_it_;
      for (Size i = 0; i < vnum; i += vsize) {
        const auto idxs = grex::load_multibyte(cit + i, vtag);
        if constexpr (is_valued) {
          const auto vals = grex::load(vit + i, vtag);
          op(idxs, vals, vtag);
        } else {
          op(idxs, vtag);
        }
      }
      if (vnum != size) {
        const auto idxs = grex::load_multibyte(cit + vnum, vtag);
        if constexpr (is_valued) {
          const auto vals = grex::load(vit + vnum, vtag);
          op(idxs, vals, grex::part_tag<vsize>(size - vnum));
        } else {
          op(idxs, grex::part_tag<vsize>(size - vnum));
        }
      }
    }

    constexpr decltype(auto) iterate(auto before, auto diagonal, auto after,
                                     AnyValuationTag auto is_valued,
                                     AnyOrderingTag auto /*is_ordered*/) const {
      auto lambda = [&](auto& op, const_iterator it) THES_ALWAYS_INLINE -> decltype(auto) {
        if constexpr (is_valued) {
          return op(it.index(), *it);
        } else {
          return op(it.index());
        }
      };

      auto j = begin();

      using Ret = thes::TypeSeq<decltype(lambda(before, j)), decltype(lambda(diagonal, j)),
                                decltype(lambda(after, j))>::Unique;

      const auto idx = col_idx();
      for (; j.index() < idx; ++j) {
        THES_APPLY_VALUED_RETURN(Ret, lambda(before, j));
      }
      {
        // The diagonal has to be stored!
        assert(j.index() == idx);
        THES_APPLY_VALUED_RETURN(Ret, lambda(diagonal, j));
        ++j;
      }
      const auto j_end = end();
      for (; j != j_end; ++j) {
        THES_APPLY_VALUED_RETURN(Ret, lambda(after, j));
      }

      THES_RETURN_EMPTY_OPTIONAL(Ret);
    }

    constexpr decltype(auto) diagonal() const {
      return (*this)[col_idx()];
    }

    [[nodiscard]] DistributedInfoCrefStorage distributed_info_storage() const {
      return dist_info_;
    }
    [[nodiscard]] DistributedInfoCrefStorage distributed_info() const
    requires(!is_shared)
    {
      return dist_info_;
    }

  private:
    template<typename TSelf>
    static std::conditional_t<std::is_const_v<TSelf>, Value, Value&> at_index(TSelf& self,
                                                                              const ColumnIdx idx) {
      const auto i = index_value(idx);
      ColumnIndexConstIter end_it = self.col_it_ + self.size();
      ColumnIndexConstIter index_it = std::lower_bound(self.col_it_, end_it, i);
      assert(index_it != end_it && *index_it == i);

      return self.entry_[index_it - self.col_it_];
    }

    static const_iterator iter_at_index(auto& self, const ColumnIdx cidx) {
      const Size i = index_value(cidx);
      ColumnIndexConstIter end_it = self.col_it_ + self.size();
      ColumnIndexConstIter index_it = std::lower_bound(self.col_it_, end_it, i);
      if (index_it == end_it || *index_it != i) [[unlikely]] {
        return end_impl(self);
      }

      return const_iterator(self.entry_, self.col_it_,
                            *thes::safe_cast<Size>(index_it - self.col_it_));
    }

    static bool contains_impl(auto& self, const ColumnIdx cidx) {
      const Size i = index_value(cidx);
      ColumnIndexConstIter end_it = self.col_it_ + self.size();
      ColumnIndexConstIter index_it = std::lower_bound(self.col_it_, end_it, i);
      return index_it != end_it && *index_it == i;
    }

    static auto begin_impl(auto& self) {
      return const_iterator{self.entry_, self.col_it_, 0};
    }
    static auto end_impl(auto& self) {
      return const_iterator{self.entry_, self.col_it_, self.size()};
    }

    [[nodiscard]] ColumnIdx col_idx() const {
      return index_convert<ColumnIdx>(row_idx_, dist_info_);
    }

    const Value* entry_;
    ColumnIndexConstIter col_it_;
    Idx row_idx_;
    Size nnz_size_;

    [[no_unique_address]] DistributedInfoCrefStorage dist_info_{};
  };

  using ConstRow = BaseConstRow<false>;
  using value_type = ConstRow;
  using ExtConstRow = BaseConstRow<true>;

private:
  template<bool tExtended>
  struct ConstIterProvider {
    using State = RowOffsetsConstIter;
    struct IterTypes
        : public thes::iter_provider::ValueTypes<BaseConstRow<tExtended>, std::ptrdiff_t> {
      using IterState = State;
    };

    static BaseConstRow<tExtended> deref(const auto& self) {
      return self.mat_->row(self.index(), self.row_off_curr_);
    }
    template<typename TSelf>
    static thes::TransferConst<TSelf, State>& state(TSelf& self) {
      return self.row_off_curr_;
    }

    static void test_if_cmp([[maybe_unused]] const auto& i1, [[maybe_unused]] const auto& i2) {
      assert(i1.mat_ == i2.mat_);
      assert(i1.row_off_begin_ == i2.row_off_begin_);
    }
  };

public:
  template<bool tExtended>
  struct BaseConstIterator
      : public thes::IteratorFacade<BaseConstIterator<tExtended>,
                                    thes::iter_provider::Map<ConstIterProvider<tExtended>>> {
    friend struct ConstIterProvider<tExtended>;
    using Idx = std::conditional_t<tExtended, ExtRowIdx, RowIdx>;

    BaseConstIterator() = default;
    BaseConstIterator(const CsrMatrixBase& mat, RowOffsetsConstIter row_off_begin,
                      RowOffsetsConstIter row_off_curr)
        : mat_{&mat}, row_off_begin_{row_off_begin}, row_off_curr_{row_off_curr} {}

    [[nodiscard]] const auto& distributed_info_storage() const {
      return mat_->distributed_info_storage();
    }

    [[nodiscard]] const CsrMatrixBase& matrix() const {
      return mat_;
    }

    [[nodiscard]] Idx index() const {
      return Idx{*thes::safe_cast<Size>(row_off_curr_ - row_off_begin_)};
    }

  private:
    const CsrMatrixBase* mat_{nullptr};
    RowOffsetsConstIter row_off_begin_{};
    RowOffsetsConstIter row_off_curr_{};
  };

  using ConstIterator = BaseConstIterator<false>;
  using const_iterator = ConstIterator;
  using ExtConstIterator = BaseConstIterator<true>;

  template<bool tExtended>
  struct BaseConstReverseIterator
      : public thes::IteratorFacade<
          BaseConstReverseIterator<tExtended>,
          thes::iter_provider::Reverse<thes::iter_provider::Map<ConstIterProvider<tExtended>>,
                                       BaseConstReverseIterator<tExtended>>> {
    friend struct ConstIterProvider<tExtended>;
    using Idx = std::conditional_t<tExtended, ExtRowIdx, RowIdx>;

    BaseConstReverseIterator() = default;
    BaseConstReverseIterator(const CsrMatrixBase& mat, RowOffsetsConstIter row_off_begin,
                             RowOffsetsConstIter row_off_curr)
        : mat_{&mat}, row_off_begin_{row_off_begin}, row_off_curr_{row_off_curr} {}

    [[nodiscard]] Idx index() const {
      return Idx{*thes::safe_cast<Size>(row_off_curr_ - row_off_begin_)};
    }

  private:
    const CsrMatrixBase* mat_{nullptr};
    RowOffsetsConstIter row_off_begin_{};
    RowOffsetsConstIter row_off_curr_{};
  };

  using ConstReverseIterator = BaseConstReverseIterator<false>;
  using const_reverse_iterator = ConstReverseIterator;
  using ExtConstReverseIterator = BaseConstReverseIterator<true>;

  struct RowWiseBuilder {
    using RowOffsetsConstIter = RowOffsets::const_iterator;

    using RowColumnIndices = thes::FlatMap<Size, Value>;

    ConstRow operator[](const RowIdx row_idx) const {
      const auto lidx = index_value(row_idx, local_index_tag, dist_info_);
      assert(lidx < row_idx_);
      assert(*thes::safe_cast<Size>(lidx + 1) < row_offs_.size());
      return row(row_idx, row_offs_.begin() + lidx);
    }

    explicit RowWiseBuilder(thes::VoidStorageRvalRef<DistributedInfo> dist_info)
        : dist_info_(std::forward<DistributedInfoStorage>(dist_info)) {}
    RowWiseBuilder()
    requires(is_shared)
    = default;

    void initialize(IsSymmetric is_symmetric) {
      row_offs_.push_back(0);
      is_symmetric_ = is_symmetric;
    }

    void initialize(Size row_num, IsSymmetric is_symmetric) {
      row_offs_.reserve(row_num + 1);
      initialize(is_symmetric);
    }

    void initialize(Size row_num, Size /*col_num*/, NonZeroSize non_zeros,
                    IsSymmetric is_symmetric) {
      row_offs_.reserve(row_num + 1);
      entries_.reserve(non_zeros);
      col_idxs_.reserve(non_zeros);
      initialize(is_symmetric);
    }

    void insert(ColumnIdx col_idx, Value value) {
      entry_map_.insert(index_value(col_idx), value);
    }

    RowWiseBuilder& operator++() {
      const auto entry_num = *thes::safe_cast<Size>(entry_map_.size());

      row_begin_ = *thes::safe_cast<NonZeroSize>(row_begin_ + entry_num);
      row_offs_.push_back(row_begin_);

      for (const auto& [key, value] : entry_map_) {
        entries_.emplace_back(value);
        col_idxs_.push_back(key);
      }
      entry_map_.clear();

      ++row_idx_;
      return *this;
    }

    void map_columns(auto column_map) {
      for (auto c_it : thes::iter_range(col_idxs_)) {
        *c_it = column_map(*c_it);
      }
    }

    void shrink_to_fit() {
      row_offs_.shrink_to_fit();
      entries_.shrink_to_fit();
      col_idxs_.shrink_to_fit();
    }

    CsrMatrixBase build(Size row_num, Size col_num) && {
      // If this is not the case, “initialize” has not been called!
      assert(row_offs_.size() > 0 && row_offs_.front() == 0);
      if constexpr (is_shared) {
        assert(row_idx_ == row_num);
      } else {
        assert(row_idx_ == dist_info_.size(local_index_tag) &&
               row_num == dist_info_.size(own_index_tag));
      }
      entries_.resize(entries_.size() + entry_pad);
      return CsrMatrixBase(row_num, col_num, std::move(row_offs_), std::move(col_idxs_),
                           std::move(entries_), is_symmetric_, std::move(dist_info_));
    }
    CsrMatrixBase build() && {
      return std::move(*this).build(row_idx_, row_idx_);
    }

    [[nodiscard]] Size index() const {
      return row_idx_;
    }
    [[nodiscard]] Size size() const {
      return entry_map_.size();
    }

    decltype(auto) row_columns() const {
      return thes::transform_range(
        [&](auto p) THES_ALWAYS_INLINE { return std::pair{ColumnIdx{p.first}, p.second}; },
        entry_map_);
    }

  private:
    ConstRow row(RowIdx index, RowOffsetsConstIter index_ptr) const {
      const NonZeroSize begin_index = index_ptr[0];
      const NonZeroSize end_index = index_ptr[1];
      return ConstRow{
        entries_.data() + begin_index,
        col_idxs_.begin() + begin_index,
        index,
        *thes::safe_cast<Size>(end_index - begin_index),
        dist_info_,
      };
    }

    IsSymmetric is_symmetric_{};

    Entries entries_{};
    RowOffsets row_offs_{};
    ColumnIndices col_idxs_{};

    Size row_idx_{0};
    NonZeroSize row_begin_{0};
    // TODO Replace with hash map and sort later?
    RowColumnIndices entry_map_{};

    [[no_unique_address]] DistributedInfoStorage dist_info_{};
  };

  template<AnyUniquenessTag TUniqueness, OptIndexTag TIndex>
  struct MultithreadPlanner {
    using Matrix = CsrMatrixBase;
    using Self = MultithreadPlanner;
    using Idx = OptIndex<is_shared, Size, TIndex>;
    static constexpr bool unique = TUniqueness::value;
    static constexpr bool extended =
      thes::StaticMap{
        thes::static_kv<own_index_tag, false>,
        thes::static_kv<local_index_tag, true>,
      }
        .get(thes::auto_tag<TIndex{}>);

    struct Data {
      explicit Data(std::size_t p_thread_num, IsSymmetric p_is_symmetric)
          : thread_num(p_thread_num), is_symmetric(p_is_symmetric), row_nums(p_thread_num),
            non_zero_nums(p_thread_num) {}

      std::size_t thread_num;
      IsSymmetric is_symmetric;
      Sizes row_nums;
      RowOffsets non_zero_nums;
    };

    struct ThreadInstance {
      using RowColumnIndices =
        std::conditional_t<unique, thes::Empty, ankerl::unordered_dense::set<Size>>;

      ThreadInstance(Data& data, std::size_t thread_index)
          : data_(data), thread_index_(thread_index) {}

      void add_column(ColumnIdx index) {
        if constexpr (unique) {
          ++non_zero_cnt_;
        } else {
          row_col_idxs_.insert(index_value(index));
        }
      }

      void operator++() {
        ++row_cnt_;
        if constexpr (!unique) {
          non_zero_cnt_ += *thes::safe_cast<NonZeroSize>(row_col_idxs_.size());
          row_col_idxs_.clear();
        }
      }

      [[nodiscard]] Idx row_index() const {
        return Idx{row_cnt_};
      }

      void finalize() {
        data_.row_nums[thread_index_] = row_cnt_;
        data_.non_zero_nums[thread_index_] = non_zero_cnt_;
      }

    private:
      Data& data_;
      std::size_t thread_index_;
      Size row_cnt_{0};
      NonZeroSize non_zero_cnt_{0};
      [[no_unique_address]] RowColumnIndices row_col_idxs_{};
    };

    MultithreadPlanner() = default;
    MultithreadPlanner(const MultithreadPlanner&) = delete;
    MultithreadPlanner(MultithreadPlanner&&) noexcept = default;
    MultithreadPlanner& operator=(const MultithreadPlanner&) = delete;
    MultithreadPlanner& operator=(MultithreadPlanner&&) noexcept = default;

    ~MultithreadPlanner() = default;

    void initialize(std::size_t thread_num, IsSymmetric is_symmetric) {
      assert(!data_.has_value());
      data_.emplace(thread_num, is_symmetric);
    }

    [[nodiscard]] ThreadInstance thread_instance(std::size_t thread_index) {
      return ThreadInstance{data(), thread_index};
    }

    [[nodiscard]] IsSymmetric is_symmetric() const {
      return data().is_symmetric;
    }

    [[nodiscard]] std::size_t thread_num() const {
      return data().thread_num;
    }

    [[nodiscard]] const Sizes& row_nums() const& {
      return data().row_nums;
    }
    [[nodiscard]] Sizes&& row_nums() && {
      return std::move(data().row_nums);
    }

    [[nodiscard]] const RowOffsets& non_zero_nums() const& {
      return data().non_zero_nums;
    }
    [[nodiscard]] RowOffsets&& non_zero_nums() && {
      return std::move(data().non_zero_nums);
    }

    [[nodiscard]] Idx row_offset(std::size_t thread_idx) const {
      const auto& row_nums = data().row_nums;
      return Idx{std::reduce(row_nums.begin(), row_nums.begin() + thread_idx, Size{0})};
    }
    [[nodiscard]] Size row_num() const {
      const auto& row_nums = data().row_nums;
      return std::reduce(row_nums.begin(), row_nums.end(), Size{0});
    }

    [[nodiscard]] NonZeroSize non_zero_offset(std::size_t thread_idx) const {
      const auto& non_zero_nums = data().non_zero_nums;
      return std::reduce(non_zero_nums.begin(), non_zero_nums.begin() + thread_idx, NonZeroSize{0});
    }

    void deallocate() {
      assert(data_.has_value());
      data_.reset();
    }

  private:
    const Data& data() const {
      assert(data_.has_value());
      return *data_;
    }
    Data& data() {
      assert(data_.has_value());
      return *data_;
    }

    std::optional<Data> data_{};
  };

  template<typename TReal, AnyUniquenessTag TUniqueness, OptIndexTag TIndex>
  struct MultithreadBuilder {
    using Real = TReal;
    using Matrix = CsrMatrixBase;
    using Self = MultithreadBuilder;
    using Idx = OptIndex<is_shared, Size, TIndex>;
    static constexpr bool unique = TUniqueness::value;

    using RowOffsetsIter = RowOffsets::iterator;
    using ColumnIndexIter = ColumnIndices::iterator;

    struct Data {
      Data(IsSymmetric p_is_symmetric, Size p_row_num, NonZeroSize p_non_zero_cnt,
           Sizes&& p_row_nums, RowOffsets&& p_non_zero_nums,
           thes::VoidStorageRvalRef<DistributedInfo> p_dist_info)
          : is_symmetric(p_is_symmetric), row_num(p_row_num), row_nums(std::move(p_row_nums)),
            non_zero_nums(std::move(p_non_zero_nums)), row_offs(p_row_num + 1),
            col_idxs(p_non_zero_cnt), entries(p_non_zero_cnt + entry_pad),
            dist_info(std::forward<DistributedInfoStorage>(p_dist_info)) {}

      IsSymmetric is_symmetric;
      Size row_num;

      Sizes row_nums;
      RowOffsets non_zero_nums;

      RowOffsets row_offs;
      ColumnIndices col_idxs;
      Entries entries;

      [[no_unique_address]] DistributedInfoStorage dist_info;
    };

    struct ThreadInstance {
      ThreadInstance(Data& data, Idx row_offset, NonZeroSize non_zero_offset)
          : non_zero_row_(non_zero_offset),
            row_ptr_(data.row_offs.begin() +
                     index_value(row_offset, local_index_tag, data.dist_info)),
            cols_begin_(data.col_idxs.begin()), vals_begin_(data.entries.data()),
            dist_info_(data.dist_info) {
        *row_ptr_ = non_zero_offset;
        ++row_ptr_;
      }

      void insert(TypedIndex<is_shared, Size, GlobalSize> auto index, Value value) {
        const Size raw_idx = index_value(index, local_index_tag, dist_info_);
        auto col_start = cols_begin_ + non_zero_row_;

        if constexpr (!unique) {
          const auto col_end = col_start + non_zero_off_;

          auto it = std::find(col_start, col_end, raw_idx);
          if (it != col_end) [[unlikely]] {
            tmp_entries_[*thes::safe_cast<Size>(it - col_start)] += Real(value);
            return;
          }
        }
        col_start[non_zero_off_] = raw_idx;
        tmp_entries_.push_back(Real(value));
        ++non_zero_off_;
      }

      void operator++() {
        *row_ptr_ = non_zero_current();
        ++row_ptr_;

        auto col_start = cols_begin_ + non_zero_row_;
        auto* val_start = tmp_entries_.data();
        std::sort(csr::IndexValueIter(NonZeroSize{0}, col_start, val_start),
                  csr::IndexValueIter(non_zero_off_, col_start, val_start));
        assert(std::is_sorted(col_start, col_start + non_zero_off_));

        std::transform(tmp_entries_.begin(), tmp_entries_.end(), vals_begin_ + non_zero_row_,
                       [](Real v) THES_ALWAYS_INLINE { return Value(v); });

        non_zero_row_ = non_zero_current();
        non_zero_off_ = 0;
        tmp_entries_.clear();
      }

      void finalize() {}

    private:
      THES_ALWAYS_INLINE NonZeroSize non_zero_current() const {
        return non_zero_row_ + non_zero_off_;
      }

      thes::DynamicArray<Real> tmp_entries_{};

      NonZeroSize non_zero_row_;
      NonZeroSize non_zero_off_{0};

      RowOffsetsIter row_ptr_;
      ColumnIndexIter cols_begin_;
      Value* vals_begin_;

      [[no_unique_address]] DistributedInfoCrefStorage dist_info_;
    };

    MultithreadBuilder() = default;
    MultithreadBuilder(const MultithreadBuilder&) = delete;
    MultithreadBuilder(MultithreadBuilder&&) noexcept = default;
    MultithreadBuilder& operator=(const MultithreadBuilder&) = delete;
    MultithreadBuilder& operator=(MultithreadBuilder&&) noexcept = default;

    ~MultithreadBuilder() = default;

    void initialize(Sizes&& row_nums, RowOffsets&& non_zero_nums, IsSymmetric is_symmetric,
                    thes::VoidStorageRvalRef<DistributedInfo> dist_info) {
      const Size part_local_num = 0;
      const Size row_cnt = std::reduce(row_nums.begin(), row_nums.end(), Size{0});
      if constexpr (!is_shared) {
        assert(row_cnt == dist_info.size(TIndex{}));
      }
      const NonZeroSize non_zero_cnt =
        std::reduce(non_zero_nums.begin(), non_zero_nums.end(), NonZeroSize{0});

      assert(!data_.has_value());
      [[maybe_unused]] Data& data =
        data_.emplace(is_symmetric, row_cnt + part_local_num, non_zero_cnt, std::move(row_nums),
                      std::move(non_zero_nums), std::forward<DistributedInfoStorage>(dist_info));
    }

    void initialize(MultithreadPlanner<TUniqueness, TIndex>&& phase1,
                    thes::VoidStorageRvalRef<DistributedInfo> dist_info) {
      initialize(std::move(phase1).row_nums(), std::move(phase1).non_zero_nums(),
                 phase1.is_symmetric(), std::forward<DistributedInfoStorage>(dist_info));
    }

    [[nodiscard]] Size row_num() const {
      assert(data_.has_value());
      return data_->row_num;
    }
    [[nodiscard]] Idx row_offset(std::size_t thread_idx) const {
      assert(data_.has_value());
      const auto& row_nums = data_->row_nums;
      const auto off = std::reduce(row_nums.begin(), row_nums.begin() + thread_idx, Size{0});
      return Idx{off};
    }
    [[nodiscard]] NonZeroSize non_zero_offset(std::size_t thread_idx) const {
      assert(data_.has_value());
      const auto& nnz_cnts = data_->non_zero_nums;
      const auto off = std::reduce(nnz_cnts.begin(), nnz_cnts.begin() + thread_idx, NonZeroSize{0});
      return off;
    }

    [[nodiscard]] ThreadInstance thread_instance(Idx row_offset, NonZeroSize non_zero_offset) {
      assert(data_.has_value());
      return ThreadInstance{*data_, row_offset, non_zero_offset};
    }
    [[nodiscard]] ThreadInstance thread_instance(std::size_t thread_idx) {
      return thread_instance(row_offset(thread_idx), non_zero_offset(thread_idx));
    }

    // This version assumes that the matrix is square!
    Matrix build() && {

      assert(data_.has_value());
      Data& data = *data_;

      if constexpr (is_shared) {
        return std::move(*this).build_impl(data.row_num, data.row_num);
      } else {
        return std::move(*this).build_impl(data.dist_info.size(own_index_tag),
                                           data.dist_info.size(local_index_tag));
      }
    }

    Matrix build(Size row_num, Size col_num) && {
      return std::move(*this).build_impl(row_num, col_num);
    }

    void deallocate() {
      assert(data_.has_value());
      data_.reset();
    }

  private:
    Matrix build_impl(Size row_num, Size col_num) && {
      assert(data_.has_value());

      Data& data = *data_;
      if constexpr (is_shared) {
        assert(data.row_num == row_num);
      } else {
        assert(data.row_num == data.dist_info.size(local_index_tag) &&
               row_num == data.dist_info.size(own_index_tag) &&
               col_num == data.dist_info.size(local_index_tag));
      }

      return Matrix(row_num, col_num, std::move(data.row_offs), std::move(data.col_idxs),
                    std::move(data.entries), data.is_symmetric,
                    std::forward<DistributedInfoStorage>(data.dist_info));
    }

    std::optional<Data> data_{};
  };

  template<AnyUniquenessTag TUniqueness, AnyIndexTag TIndex>
  static auto multithread_planner(TUniqueness /*uniqueness*/, TIndex /*index*/) {
    return MultithreadPlanner<TUniqueness, TIndex>{};
  }
  template<AnyUniquenessTag TUniqueness>
  requires(is_shared)
  static auto multithread_planner(TUniqueness /*uniqueness*/) {
    return MultithreadPlanner<TUniqueness, OwnIndexTag>{};
  }

  template<typename TReal, AnyUniquenessTag TUniqueness, AnyIndexTag TIndex>
  static auto multithread_builder(thes::TypeTag<TReal> /*real*/, TUniqueness /*uniqueness*/,
                                  TIndex /*index*/) {
    return MultithreadBuilder<TReal, TUniqueness, TIndex>{};
  }
  template<typename TReal, AnyUniquenessTag TUniqueness>
  requires(is_shared)
  static auto multithread_builder(thes::TypeTag<TReal> /*real*/, TUniqueness /*uniqueness*/) {
    return MultithreadBuilder<TReal, TUniqueness, OwnIndexTag>{};
  }

  CsrMatrixBase(IsSymmetric is_symmetric, thes::VoidStorageRvalRef<DistributedInfo> dist_info)
      : is_symmetric_(is_symmetric), dist_info_(std::forward<DistributedInfoStorage>(dist_info)) {}
  explicit CsrMatrixBase(IsSymmetric is_symmetric)
  requires(is_shared)
      : is_symmetric_(is_symmetric) {}

  explicit CsrMatrixBase(Size row_num, Size col_num, RowOffsets&& row_offs,
                         ColumnIndices&& col_idxs, Entries&& entries, IsSymmetric is_symmetric,
                         thes::VoidStorageRvalRef<DistributedInfo> dist_info)
      : is_symmetric_(is_symmetric), row_num_(row_num), col_num_(col_num),
        row_offs_(std::forward<RowOffsets>(row_offs)),
        col_idxs_(std::forward<ColumnIndices>(col_idxs)), entries_(std::forward<Entries>(entries)),
        dist_info_(std::forward<DistributedInfoStorage>(dist_info)) {}

  explicit CsrMatrixBase(Size row_num, Size col_num, RowOffsets&& row_offs,
                         ColumnIndices&& col_idxs, Entries&& entries, IsSymmetric is_symmetric)
  requires(is_shared)
      : row_num_(row_num), col_num_(col_num), row_offs_(std::forward<RowOffsets>(row_offs)),
        col_idxs_(std::forward<ColumnIndices>(col_idxs)), entries_(std::forward<Entries>(entries)),
        is_symmetric_(is_symmetric) {}

  [[nodiscard]] ConstRow operator[](const RowIdx idx) const {
    return row(idx, index_begin() + index_value(idx));
  }

  [[nodiscard]] ConstIterator begin() const {
    return ConstIterator{*this, index_begin(), index_begin()};
  }
  [[nodiscard]] ConstIterator end() const {
    return ConstIterator{*this, index_begin(), index_end()};
  }

  [[nodiscard]] ConstReverseIterator rbegin() const {
    return ConstReverseIterator{*this, index_begin(), index_end()};
  }
  [[nodiscard]] ConstReverseIterator rend() const {
    return ConstReverseIterator{*this, index_begin(), index_begin()};
  }

  [[nodiscard]] ExtConstIterator ext_begin() const {
    return ExtConstIterator{*this, row_offs_.begin(), row_offs_.begin()};
  }
  [[nodiscard]] ExtConstIterator ext_end() const {
    return ExtConstIterator{*this, row_offs_.begin(), row_offs_.begin() + ext_row_num()};
  }
  [[nodiscard]] auto ext_range() const {
    return thes::value_range(ext_begin(), ext_end());
  }

  [[nodiscard]] ConstIterator iter_at(const RowIdx idx) const {
    return ConstIterator{*this, index_begin(), index_begin() + idx};
  }

  [[nodiscard]] Value diagonal_at(const ExtRowIdx index) const {
    return diagonal_at_impl(index, row_offs_, col_idxs_, entries_);
  }
  [[nodiscard]] Value entry_at(const ExtRowIdx row, const ColumnIdx col) const {
    return entry_at_impl(row, col, row_offs_, col_idxs_, entries_);
  }

  [[nodiscard]] IsSymmetric is_symmetric() const {
    return is_symmetric_;
  }

  [[nodiscard]] Size row_num() const {
    return row_num_;
  }
  [[nodiscard]] Size ext_row_num() const {
    if constexpr (is_shared) {
      return row_num();
    } else {
      assert(col_num_ == dist_info_.size(local_index_tag));
      return dist_info_.size(local_index_tag);
    }
  }
  [[nodiscard]] Size column_num() const {
    return col_num_;
  }
  [[nodiscard]] NonZeroSize non_zero_num() const {
    assert(entries_.size() >= entry_pad);
    return *thes::safe_cast<NonZeroSize>(entries_.size() - entry_pad);
  }

  const DistributedInfoStorage& distributed_info_storage() const {
    return dist_info_;
  }
  const DistributedInfoStorage& distributed_info() const
  requires(!is_shared)
  {
    return distributed_info_storage();
  }

  auto row_offsets() const {
    return row_offs_.sub_range(0, row_num_);
  }
  const RowOffsets& row_offsets_extended() const {
    return row_offs_;
  }
  const ColumnIndices& column_indices() const {
    return col_idxs_;
  }
  std::span<const Value> entries() const {
    assert(entries_.size() >= entry_pad);
    return std::span{entries_.data(), entries_.size() - entry_pad};
  }

private:
  template<bool tExtended>
  BaseConstRow<tExtended> row_impl(std::conditional_t<tExtended, ExtRowIdx, RowIdx> row_idx,
                                   RowOffsetsConstIter index_ptr) const {
    if constexpr (tExtended) {
      assert(index_value(row_idx) < ext_row_num());
    } else {
      assert(index_value(row_idx) < row_num_);
    }
    const NonZeroSize begin_index = index_ptr[0];
    const NonZeroSize end_index = index_ptr[1];
    return {
      entries_.data() + begin_index,
      col_idxs_.begin() + begin_index,
      row_idx,
      *thes::safe_cast<Size>(end_index - begin_index),
      dist_info_,
    };
  }

  ConstRow row(RowIdx row_idx, RowOffsetsConstIter index_ptr) const {
    return row_impl<false>(row_idx, index_ptr);
  }
  ExtConstRow row(ExtRowIdx row_idx, RowOffsetsConstIter index_ptr) const
  requires(!is_shared)
  {
    return row_impl<true>(row_idx, index_ptr);
  }

  [[nodiscard]] static auto index_begin_impl(auto& self) {
    if constexpr (is_shared) {
      return self.row_offs_.begin();
    } else {
      return self.row_offs_.begin() + self.dist_info_.own_begin(local_index_tag);
    }
  }

  [[nodiscard]] RowOffsetsConstIter index_begin() const {
    return index_begin_impl(*this);
  }
  [[nodiscard]] RowOffsetsIter index_begin() {
    return index_begin_impl(*this);
  }
  [[nodiscard]] RowOffsetsConstIter index_end() const {
    return index_begin() + row_num_;
  }
  [[nodiscard]] RowOffsetsIter index_end() {
    return index_begin() + row_num_;
  }

  static auto row_idxs(const ExtRowIdx row, const RowOffsets& row_offs) {
    const Size raw_row = index_value(row);
    assert(raw_row + 1 < row_offs.size());

    const NonZeroSize begin_index = row_offs[raw_row];
    const NonZeroSize end_index = row_offs[raw_row + 1];
    assert(end_index >= begin_index);

    return std::make_pair(begin_index, end_index);
  }

  static Value entry_at_impl(const ExtRowIdx row, const ExtRowIdx col, const RowOffsets& row_offs,
                             const ColumnIndices& col_idxs, const Entries& entries) {
    const auto [begin_index, end_index] = row_idxs(row, row_offs);
    const Size raw_col = index_value(col);

    ColumnIndexConstIter col_begin = col_idxs.begin();
    ColumnIndexConstIter row_col_end = col_begin + end_index;
    ColumnIndexConstIter col_it = std::lower_bound(col_begin + begin_index, row_col_end, raw_col);

    if (col_it != row_col_end && *col_it == raw_col) [[likely]] {
      return entries[*thes::safe_cast<NonZeroSize>(col_it - col_begin)];
    }
    return 0;
  }

  static Value diagonal_at_impl(const ExtRowIdx index, const RowOffsets& row_offs,
                                const ColumnIndices& col_idxs, const Entries& entries) {
    const auto v = entry_at_impl(index, index, row_offs, col_idxs, entries);
    assert(v != 0);
    return v;
  }

  IsSymmetric is_symmetric_;
  Size row_num_{0};
  Size col_num_{0};

  RowOffsets row_offs_{0};
  ColumnIndices col_idxs_{};
  Entries entries_{};

  [[no_unique_address]] DistributedInfoStorage dist_info_{};
};

template<typename TValue, typename TSizeByte, typename TNonZeroSizeByte,
         typename TAlloc = thes::HugePagesAllocator<TValue>>
struct CsrMatrixDefaultDefs {
  using Value = TValue;
  using SizeByte = TSizeByte;
  using NonZeroSizeByte = TNonZeroSizeByte;
  using Alloc = TAlloc;
};

template<typename TValue, typename TSizeByte, typename TNonZeroSizeByte,
         typename TAlloc = thes::HugePagesAllocator<TValue>>
using CsrMatrix = CsrMatrixBase<CsrMatrixDefaultDefs<TValue, TSizeByte, TNonZeroSizeByte, TAlloc>>;
} // namespace lineal

#endif // INCLUDE_LINEAL_TENSOR_DYNAMIC_CSR_SHARED_HPP
