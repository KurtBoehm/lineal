// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_TENSOR_DYNAMIC_STENCIL_MATRIX_SHARED_HPP
#define INCLUDE_LINEAL_TENSOR_DYNAMIC_STENCIL_MATRIX_SHARED_HPP

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <exception>
#include <functional>
#include <string>
#include <type_traits>
#include <utility>

#include "thesauros/algorithms.hpp"
#include "thesauros/format.hpp"
#include "thesauros/iterator.hpp"
#include "thesauros/macropolis/inlining.hpp"
#include "thesauros/types.hpp"
#include "thesauros/utility.hpp"

#include "lineal/base.hpp"
#include "lineal/component-wise.hpp"
#include "lineal/parallel.hpp"
#include "lineal/tensor/dynamic/stencil/matrix/algorithm.hpp"
#include "lineal/vectorization.hpp"

namespace lineal {
template<typename TValuator>
struct AdjacentStencilMatrix : public SharedMatrixBase {
  using Valuator = std::decay_t<TValuator>;
  using Real = Valuator::Real;
  using Value = Real;
  using SizeByte = Valuator::SizeByte;
  using Size = SizeByte::Unsigned;
  using NonZeroSizeByte = Valuator::NonZeroSizeByte;
  using NonZeroSize = NonZeroSizeByte::Unsigned;

  using RawDistributedInfo = Valuator::RawDistributedInfo;
  using DistributedInfo = std::decay_t<RawDistributedInfo>;
  using DistributedInfoStorage = thes::VoidStorage<DistributedInfo>;
  static constexpr bool is_shared = std::is_void_v<DistributedInfo>;
  static constexpr thes::Tuple exec_constraints{GeometryPreservingVectorizationConstraint{}};

  static_assert(std::same_as<Size, LocalSizeOf<DistributedInfo, Size>>);
  using GlobalSize = GlobalSizeOf<DistributedInfo, Size>;
  using RowIdx = OptOwnIndex<is_shared, Size>;
  using ExtRowIdx = OptLocalIndex<is_shared, Size>;
  using ColumnIdx = ExtRowIdx;

  static constexpr std::size_t dimension_num = Valuator::dimension_num;
  static constexpr AdjacentStencilNnzPattern nnz_pattern{};

  using GlobalPosition = std::array<GlobalSize, dimension_num>;
  template<TypedIndex<is_shared, Size, GlobalSize> TIdx>
  using IdxPos = thes::IndexPosition<TIdx, GlobalPosition>;

  struct TooSmallOverlap : public std::exception {
    TooSmallOverlap(Size before_overlap, Size after_overlap, Size required_overlap)
        : msg_(fmt::format("The overlaps are {} and {}, but have to be at least {} (or 0 at the "
                           "beginning and end)!",
                           before_overlap, after_overlap, required_overlap)) {}

    [[nodiscard]] const char* what() const noexcept override {
      return msg_.c_str();
    }

  private:
    std::string msg_;
  };

  template<bool tExtended>
  struct BaseConstRow {
    using Idx = std::conditional_t<tExtended, ExtRowIdx, RowIdx>;

    BaseConstRow(Idx index, GlobalPosition pos, const Valuator& valuator)
        : row_idx_(index), global_pos_(pos), valuator_(valuator) {}
    BaseConstRow(Idx index, const Valuator& valuator)
        : row_idx_(index), global_pos_(compute_pos(row_idx_, valuator)), valuator_(valuator) {}

    [[nodiscard]] THES_ALWAYS_INLINE Idx index() const {
      return row_idx_;
    }
    [[nodiscard]] THES_ALWAYS_INLINE ExtRowIdx ext_index() const {
      return index_convert<ExtRowIdx>(row_idx_, dinfo());
    }

    [[nodiscard]] THES_ALWAYS_INLINE GlobalPosition global_position() const {
      return global_pos_;
    }

    [[nodiscard]] THES_ALWAYS_INLINE const Valuator& valuator() const {
      return valuator_;
    }

    // TODO Should zeros be ignored?
    // That depends on the use-case, unfortunately: When determining the number of entries in a row
    // to store e.g. edge properties, it *might* be desirable to incorporate zeros, but otherwise
    // this does not appear to make much sense.
    [[nodiscard]] Size offdiagonal_num() const {
      decltype(auto) sys_info = valuator_.info();
      Size num = 0;
      thes::star::iota<0, dimension_num> | thes::star::for_each([&](auto i) {
        const auto ci = std::get<i>(global_pos_);
        num += Size{ci > 0} + Size{ci + 1 < sys_info.axis_size(i)};
      });
      return num;
    }

    THES_ALWAYS_INLINE decltype(auto) iterate(auto before, auto diagonal, auto after,
                                              AnyValuationTag auto is_valued,
                                              AnyOrderingTag auto is_ordered) const {
      return stencil::iterate(
        ext_index(), global_pos_,
        [&](auto base, auto off, auto op) THES_ALWAYS_INLINE {
          if constexpr (tExtended && !is_shared) {
            if constexpr (std::same_as<decltype(op), std::minus<>>) {
              return index_value(base) >= off;
            }
            if constexpr (std::same_as<decltype(op), std::plus<>>) {
              return index_value(base) < dinfo().size(local_index_tag) - off;
            }
          } else {
            return true;
          }
        },
        valuator_, std::move(before), std::move(diagonal), std::move(after), is_valued, is_ordered);
    }

    THES_ALWAYS_INLINE decltype(auto) iterate(auto op, AnyValuationTag auto is_valued,
                                              AnyOrderingTag auto is_ordered) const {
      return iterate(op, op, op, is_valued, is_ordered);
    }

    [[nodiscard]] THES_ALWAYS_INLINE bool banded_valid(grex::VectorTag auto tag) const {
      constexpr auto last = dimension_num - 1;
      return thes::star::get_at<last>(global_pos_) + tag.part() <=
             valuator().info().axis_size(thes::auto_tag<last>);
    }

    THES_ALWAYS_INLINE void banded_iterate(auto before, auto diagonal, auto after,
                                           AnyOrderingTag auto is_ordered,
                                           grex::VectorTag auto tag) const
    requires(!tExtended)
    {
      stencil::banded_iterate(ext_index(), global_pos_, valuator_, std::move(before),
                              std::move(diagonal), std::move(after), is_ordered, tag);
    }

    THES_ALWAYS_INLINE void banded_iterate(auto op, AnyOrderingTag auto is_ordered,
                                           grex::VectorTag auto tag) const
    requires(!tExtended)
    {
      assert(banded_valid(tag));
      auto wrapop = [&](auto val, auto off, auto /*dim*/)
                      THES_ALWAYS_INLINE { return op(val, std::move(off)); };
      banded_iterate(wrapop, op, wrapop, is_ordered, tag);
    }

    [[nodiscard]] Real diagonal() const {
      return stencil::diagonal(ext_index(), global_pos_, valuator_);
    }

    [[nodiscard]] Size
    offdiagonal_index_within(const TypedIndex<is_shared, Size, GlobalSize> auto idx) const {
      const ColumnIdx i = index_convert(idx, local_index_tag, dinfo());
      return stencil::offdiagonal_index_within(ext_index(), global_pos_, valuator_, i,
                                               [](auto /*c*/) THES_ALWAYS_INLINE { return true; });
    }

    const DistributedInfoStorage& distributed_info_storage() const {
      return dinfo();
    }

  private:
    [[nodiscard]] decltype(auto) dinfo() const {
      return valuator_.distributed_info_storage();
    }

    Idx row_idx_;
    GlobalPosition global_pos_;
    const Valuator& valuator_;
  };

  using ConstRow = BaseConstRow<false>;
  using ExtConstRow = BaseConstRow<true>;

private:
  template<bool tExtended>
  struct ConstIterProvider {
    using State = Size;
    using Idx = std::conditional_t<tExtended, ExtRowIdx, RowIdx>;
    using Value = BaseConstRow<tExtended>;

    struct IterTypes : public thes::iter_provider::ValueTypes<Value, std::ptrdiff_t> {
      using IterState = State;
    };

    static Value deref(const auto& self) {
      return Value{Idx{self.index_}, *self.valuator_};
    }
    template<typename TSelf>
    static thes::TransferConst<TSelf, State>& state(TSelf& self) {
      return self.index_;
    }

    static void test_if_cmp([[maybe_unused]] const auto& i1, [[maybe_unused]] const auto& i2) {
      assert(i1.valuator_ == i2.valuator_);
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
    BaseConstIterator(Idx index, const Valuator& valuator)
        : index_(index_value(index)), valuator_(&valuator) {}

  private:
    Size index_{};
    const Valuator* valuator_{};
  };

  using ConstIterator = BaseConstIterator<false>;
  using const_iterator = ConstIterator;
  using ExtConstIterator = BaseConstIterator<true>;

  explicit AdjacentStencilMatrix(TValuator&& valuator)
      : valuator_(std::forward<TValuator>(valuator)) {
    if constexpr (!is_shared) {
      assert(valuator_.data_size() == material_range(info(), dinfo()).size());
      const auto size_before = dinfo().size(before_tag);
      const auto size_after = dinfo().size(after_tag);
      const auto size_req = valuator_.info().after_size(thes::index_tag<0>);
      if ((size_before != 0 && size_before < size_req) ||
          (size_after != 0 && size_after < size_req)) [[unlikely]] {
        throw TooSmallOverlap(size_before, size_after, size_req);
      }
    }
  }

  ConstRow operator[](TypedIndex<is_shared, Size, GlobalSize> auto index) const {
    return ConstRow(index_convert(index, own_index_tag, dinfo()), valuator_);
  }
  template<TypedIndex<is_shared, Size, GlobalSize> TIdx>
  auto operator[](IdxPos<TIdx> idxpos) const {
    return ConstRow(index_convert(idxpos.index, own_index_tag, dinfo()), idxpos.position,
                    valuator_);
  }

  Real diagonal_at(ExtRowIdx idx) const {
    return stencil::diagonal(idx, compute_pos(idx, valuator_), valuator_);
  }
  Real entry_at(ExtRowIdx row, ExtRowIdx col) const {
    return stencil::offdiagonal(row, compute_pos(row, valuator_), valuator_, col).value_or(Real{0});
  }

  static constexpr IsSymmetric is_symmetric() {
    return IsSymmetric{Valuator::is_symmetric};
  }

  ConstIterator begin() const {
    return ConstIterator{RowIdx{0}, valuator_};
  }
  ConstIterator end() const {
    return ConstIterator{RowIdx{row_num()}, valuator_};
  }
  ConstIterator iter_at(RowIdx idx) const {
    return ConstIterator{idx, valuator_};
  }

  ExtConstIterator ext_begin() const {
    return ExtConstIterator{ExtRowIdx{0}, valuator_};
  }
  ExtConstIterator ext_end() const {
    return ExtConstIterator{ExtRowIdx{ext_row_num()}, valuator_};
  }
  auto ext_range() const {
    return thes::value_range(ext_begin(), ext_end());
  }
  ExtConstIterator ext_iter_at(ExtRowIdx idx) const {
    return ExtConstIterator{idx, valuator_};
  }

  const DistributedInfoStorage& distributed_info() const
  requires(!is_shared)
  {
    return dinfo();
  }
  const Valuator& valuator() const {
    return valuator_;
  }

  [[nodiscard]] Size row_num() const {
    if constexpr (is_shared) {
      return info().total_size();
    } else {
      return dinfo().size(own_index_tag);
    }
  }
  [[nodiscard]] Size ext_row_num() const {
    if constexpr (is_shared) {
      return info().total_size();
    } else {
      return dinfo().size(local_index_tag);
    }
  }
  [[nodiscard]] Size column_num() const {
    return ext_row_num();
  }

  auto axis_range(thes::AnyIndexTag auto idx) const {
    if constexpr (is_shared) {
      return thes::range(valuator_.info().axis_size(idx));
    } else {
      return dinfo().axis_range(idx);
    }
  }
  decltype(auto) geometry() const {
    return valuator_.info();
  }

private:
  [[nodiscard]] THES_ALWAYS_INLINE static GlobalPosition
  compute_pos(TypedIndex<is_shared, Size, GlobalSize> auto idx, const Valuator& valuator) {
    return valuator.info().index_to_pos(
      index_value(idx, global_index_tag, valuator.distributed_info_storage()));
  }

  [[nodiscard]] Size size() const {
    return valuator_.size();
  }
  [[nodiscard]] decltype(auto) info() const {
    return valuator_.info();
  }
  [[nodiscard]] decltype(auto) dinfo() const {
    return valuator_.distributed_info_storage();
  }

  const auto* own_data() const {
    return valuator_.data() + dinfo().before_size();
  }
  auto* own_data() {
    return valuator_.data() + dinfo().before_size();
  }

  TValuator valuator_;
};
} // namespace lineal

#endif // INCLUDE_LINEAL_TENSOR_DYNAMIC_STENCIL_MATRIX_SHARED_HPP
