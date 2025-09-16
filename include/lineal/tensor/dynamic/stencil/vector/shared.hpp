// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_TENSOR_DYNAMIC_STENCIL_VECTOR_SHARED_HPP
#define INCLUDE_LINEAL_TENSOR_DYNAMIC_STENCIL_VECTOR_SHARED_HPP

#include <array>
#include <cassert>
#include <cstddef>
#include <type_traits>

#include "thesauros/algorithms.hpp"
#include "thesauros/iterator.hpp"
#include "thesauros/macropolis/inlining.hpp"
#include "thesauros/types.hpp"
#include "thesauros/utility.hpp"

#include "lineal/base.hpp"
#include "lineal/component-wise/exec-constraint.hpp"
#include "lineal/parallel.hpp"
#include "lineal/tensor/dynamic/stencil/def.hpp"
#include "lineal/vectorization.hpp"

namespace lineal {
template<typename TValuator>
struct AdjacentStencilVector : public SharedVectorBase {
  using Valuator = std::decay_t<TValuator>;
  using Real = Valuator::Real;
  using Value = Real;
  using SizeByte = Valuator::SizeByte;
  using Size = SizeByte::Unsigned;
  using RealSize = thes::Union<Size, grex::FloatSize<Real>>;
  using SystemInfo = Valuator::SystemInfo;

  using RawDistributedInfo = Valuator::RawDistributedInfo;
  using DistributedInfo = std::decay_t<RawDistributedInfo>;
  using DistributedInfoStorage = thes::VoidStorage<DistributedInfo>;
  using DistributedInfoCrefStorage = thes::VoidStorageConstLvalRef<DistributedInfo>;
  static constexpr bool is_shared = std::is_void_v<DistributedInfo>;
  static constexpr thes::Tuple exec_constraints{GeometryPreservingVectorizationConstraint{}};

  using Index = OptOwnIndex<is_shared, Size>;
  using ExtIndex = OptLocalIndex<is_shared, Size>;

  static constexpr std::size_t dimension_num = Valuator::dimension_num;

  using GlobalSize = GlobalSizeOf<DistributedInfo, Size>;
  using GlobalPosition = std::array<GlobalSize, dimension_num>;
  template<TypedIndex<is_shared, Size, GlobalSize> TIdx>
  using IdxPos = thes::IndexPosition<TIdx, GlobalPosition>;

private:
  template<bool tExtended>
  struct ConstIterProvider {
    using Value = Real;
    using Idx = std::conditional_t<tExtended, ExtIndex, Index>;
    using State = Size;

    struct IterTypes : public thes::iter_provider::ValueTypes<Value, std::ptrdiff_t> {
      using IterState = State;
    };

    static Value deref(const auto& self) {
      return compute_impl(*self.valuator_, Idx{self.idx_}, grex::scalar_tag);
    }
    template<typename TSelf>
    static thes::TransferConst<TSelf, State>& state(TSelf& self) {
      return self.idx_;
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
    using Idx = std::conditional_t<tExtended, ExtIndex, Index>;

    BaseConstIterator() = default;
    explicit BaseConstIterator(const Valuator& valuator, Idx idx)
        : valuator_(&valuator), idx_(index_value(idx)) {}

    auto compute(grex::AnyTag auto tag) const {
      return compute_impl(*valuator_, Idx{idx_}, tag);
    }

  private:
    const Valuator* valuator_{};
    Size idx_{};
  };

  using ConstIterator = BaseConstIterator<false>;
  using const_iterator = ConstIterator;
  using ExtConstIterator = BaseConstIterator<true>;

  explicit AdjacentStencilVector(TValuator&& valuator)
      : valuator_(std::forward<TValuator>(valuator)) {}

  [[nodiscard]] Real operator[](TypedIndex<is_shared, Size, GlobalSize> auto idx) const {
    return compute_impl(valuator_, idx, grex::scalar_tag);
  }
  template<TypedIndex<is_shared, Size, GlobalSize> TIdx>
  [[nodiscard]] Real operator[](IdxPos<TIdx> idx) const {
    return compute_impl(valuator_, idx.index, idx.position, grex::scalar_tag);
  }

  auto compute(TypedIndex<is_shared, Size, GlobalSize> auto idx, grex::AnyTag auto tag) const {
    return compute_impl(valuator_, idx, tag);
  }
  template<TypedIndex<is_shared, Size, GlobalSize> TIdx>
  auto compute(IdxPos<TIdx> idx, grex::AnyTag auto tag) const {
    return compute_impl(valuator_, idx.index, idx.position, tag);
  }

  const DistributedInfoStorage& distributed_info() const {
    return dinfo();
  }

  Size size() const {
    if constexpr (is_shared) {
      return valuator_.size();
    } else {
      return dinfo().size(own_index_tag);
    }
  }
  Size ext_size() const {
    if constexpr (is_shared) {
      return valuator_.size();
    } else {
      return dinfo().size(local_index_tag);
    }
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

  const auto* data() const {
    return valuator_.data();
  }
  auto* data() {
    return valuator_.data();
  }

  ConstIterator begin() const {
    return ConstIterator(valuator_, Index{0});
  }
  ConstIterator end() const {
    return ConstIterator(valuator_, Index{size()});
  }
  ConstIterator iter_at(Index idx) const {
    return ConstIterator{valuator_, idx};
  }

  ExtConstIterator ext_begin() const {
    return ExtConstIterator{valuator_, ExtIndex{0}};
  }
  ExtConstIterator ext_end() const {
    return ExtConstIterator{valuator_, ExtIndex{ext_size()}};
  }
  ExtConstIterator ext_iter_at(ExtIndex idx) const {
    return ExtConstIterator{valuator_, idx};
  }

private:
  static Real compute_impl(const Valuator& valuator,
                           const TypedIndex<is_shared, Size, GlobalSize> auto idx, auto gpos,
                           grex::OptValuedScalarTag<Real> auto /*tag*/) {
    decltype(auto) dist_info = valuator.distributed_info_storage();
    const auto lidx = index_value(idx, local_index_tag, dist_info);

    const SystemInfo& info = valuator.info();
    Real sum = 0;
    const auto cell_value = valuator.cell_value(*(valuator.begin() + lidx), grex::scalar_tag);

    thes::star::iota<0, dimension_num> | thes::star::only_range<Valuator::non_zero_borders> |
      thes::star::for_each([&](thes::AnyIndexTag auto j) THES_ALWAYS_INLINE {
        const Size axis_index = thes::star::get_at(gpos, j);
        if (axis_index == 0) {
          sum +=
            valuator.template border_rhs_summand<j, AxisSide::START>(cell_value, grex::scalar_tag);
        }
        if (axis_index + 1 == info.axis_size(j)) {
          sum +=
            valuator.template border_rhs_summand<j, AxisSide::END>(cell_value, grex::scalar_tag);
        }
      });

    return sum;
  }

  template<grex::OptValuedVectorTag<Real> TTag>
  static grex::Vector<Real, TTag::size>
  compute_impl(const Valuator& valuator, const TypedIndex<is_shared, Size, GlobalSize> auto idx,
               auto gpos, TTag tag) {
    static constexpr std::size_t size = TTag::size;
    decltype(auto) dist_info = valuator.distributed_info_storage();
    const auto lidx = index_value(idx, local_index_tag, dist_info);

    auto real_tag = tag.instantiate(grex::type_tag<Real>);
    const auto cell_info = (valuator.begin() + lidx).load(tag);
    const auto cell_value = valuator.cell_value(cell_info, real_tag);
    const SystemInfo& info = valuator.info();
    auto sum = grex::Vector<Real, size>::zeros();

    thes::star::iota<0, dimension_num> | thes::star::only_range<Valuator::non_zero_borders> |
      thes::star::for_each([&](thes::AnyIndexTag auto j) THES_ALWAYS_INLINE {
        using Vec = grex::Vector<RealSize, size>;
        const Vec axis_index{thes::star::get_at(gpos, j)};

        // assumption: cell_value = 0 → border_rhs_summand = 0
        const auto start_mask = grex::convert_safe<Real>(axis_index == Vec::zeros());
        auto start_val =
          valuator.template border_rhs_summand<j, AxisSide::START>(cell_value, real_tag);
        sum = grex::mask_add(start_mask, sum, start_val);

        const auto end_mask =
          grex::convert_safe<Real>(axis_index + Vec{1} == Vec(info.axis_size(j)));
        auto end_val = valuator.template border_rhs_summand<j, AxisSide::END>(cell_value, real_tag);
        sum = grex::mask_add(end_mask, sum, end_val);
      });

    return sum;
  }
  static auto compute_impl(const Valuator& valuator,
                           const TypedIndex<is_shared, Size, GlobalSize> auto idx,
                           grex::AnyTag auto tag) {
    decltype(auto) dist_info = valuator.distributed_info_storage();
    const SystemInfo& info = valuator.info();

    const auto gidx = index_value(idx, global_index_tag, dist_info);
    const auto gpos_fun = [&](auto j) {
      if constexpr (grex::AnyScalarTag<decltype(tag)>) {
        return info.index_to_axis_index(gidx, j);
      } else {
        return info.index_to_axis_index(gidx, tag, thes::type_tag<RealSize>, j);
      }
    };

    return compute_impl(valuator, idx, thes::star::index_transform<dimension_num>(gpos_fun), tag);
  }

  [[nodiscard]] decltype(auto) dinfo() const {
    return valuator_.distributed_info_storage();
  }

  TValuator valuator_;
};
} // namespace lineal

#endif // INCLUDE_LINEAL_TENSOR_DYNAMIC_STENCIL_VECTOR_SHARED_HPP
