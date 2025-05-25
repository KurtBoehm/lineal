// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_TENSOR_DYNAMIC_STENCIL_VALUATOR_STATIONARY_DIFFUSION_VALUES_HPP
#define INCLUDE_LINEAL_TENSOR_DYNAMIC_STENCIL_VALUATOR_STATIONARY_DIFFUSION_VALUES_HPP

#include <array>
#include <cassert>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

#include "thesauros/macropolis.hpp"
#include "thesauros/memory.hpp"
#include "thesauros/static-ranges.hpp"
#include "thesauros/types.hpp"

#include "lineal/base.hpp"
#include "lineal/parallel.hpp"
#include "lineal/tensor/dynamic/stencil/def.hpp"
#include "lineal/tensor/dynamic/stencil/storage/cell-storage.hpp"
#include "lineal/vectorization.hpp"

namespace lineal {
template<typename TDefs, AnyZeroHandlingTag TZeroHandling, OptDistributedInfo TDistInfo,
         typename TByteAlloc = thes::HugePagesAllocator<std::byte>>
struct StationaryDiffusionValuesValuator : public ValuatorBase, public TDefs {
  using Real = TDefs::Real;
  using Size = TDefs::Size;

  static constexpr bool zero_to_one = TZeroHandling::value;
  static constexpr std::size_t dimension_num = TDefs::dimension_num;
  static constexpr std::size_t max_material_num = TDefs::max_material_num;
  static constexpr std::size_t flow_axis = TDefs::flow_axis;

  using RawSystemInfo = TDefs::SystemInfo;
  using SystemInfo = std::decay_t<RawSystemInfo>;
  using CellInfo = std::array<Real, dimension_num>;
  using LookupWright = TDefs::LookupWright;

  using RawDistributedInfo = TDistInfo;
  using DistributedInfo = std::decay_t<TDistInfo>;
  using DistributedInfoStorage = thes::VoidStorage<TDistInfo>;
  static constexpr bool is_shared = std::is_void_v<TDistInfo>;

  static constexpr std::array non_zero_borders{flow_axis};
  static constexpr bool is_symmetric = LookupWright::is_symmetric;

  static_assert(is_symmetric, "Only symmetric lookup wrights are allowed!");

  using Allocator = std::allocator_traits<TByteAlloc>::template rebind_alloc<Real>;
  using AxisStorage = CellStorage<Real, Allocator>;
  using Storage = std::array<AxisStorage, dimension_num>;
  using DiffusionComputer =
    decltype(std::declval<const LookupWright&>().diffusion_factor_computer());
  using DataOffsetStorage = std::conditional_t<is_shared, thes::ValueTag<Size, 0>, Size>;

  struct Iter {
    using Ptrs = std::array<const Real*, dimension_num>;

    explicit constexpr Iter(Ptrs begins, std::ptrdiff_t pos) : begins_(begins), pos_(pos) {}

    auto operator*() const {
      return begins_ | thes::star::transform([this](const Real* ptr) { return ptr[pos_]; }) |
             thes::star::to_array;
    }
    auto load_ext(grex::AnyTag auto tag) const {
      return begins_ | thes::star::transform([this, tag](const Real* ptr) {
               return grex::load_ptr_extended(ptr + pos_, tag);
             }) |
             thes::star::to_array;
    }
    auto load(grex::AnyTag auto tag) const {
      return begins_ | thes::star::transform([this, tag](const Real* ptr) {
               return grex::load_ptr(ptr + pos_, tag);
             }) |
             thes::star::to_array;
    }

    Iter operator+(std::ptrdiff_t off) const {
      return Iter{begins_, pos_ + off};
    }
    Iter operator-(std::ptrdiff_t off) const {
      return Iter{begins_, pos_ - off};
    }

  private:
    Ptrs begins_;
    std::ptrdiff_t pos_;
  };

  explicit StationaryDiffusionValuesValuator(RawSystemInfo&& system_info, Storage&& storage,
                                             thes::VoidStorageRvalRef<TDistInfo> dist_info = {},
                                             const LookupWright& maker = LookupWright{})
      : sys_info_(std::forward<RawSystemInfo>(system_info)), storage_(std::move(storage)),
        diff_(maker.diffusion_factor_computer()),
        dist_info_(std::forward<DistributedInfoStorage>(dist_info)) {
    assert(storage | thes::star::transform([](const AxisStorage& s) { return s.size(); }) |
           thes::star::has_unique_value);

    if constexpr (is_shared) {
      assert(sys_info_.total_size() == std::get<0>(storage).size());
    } else {
      const auto range = material_range(sys_info_, dist_info_);
      assert(range.size() == std::get<0>(storage).size());
      data_offset_ = dist_info_.local_begin(global_index_tag) - range.begin_value();
    }
  }

  THES_ALWAYS_INLINE auto cell_value(auto info, grex::AnyTag auto tag) const {
    return info | thes::star::transform([tag](auto axinfo) { return tag.mask(axinfo); }) |
           thes::star::to_array;
  }

  THES_ALWAYS_INLINE auto diagonal_base(auto /*cell_value*/, auto tag) const {
    return grex::constant<Real>(0, tag);
  }

  template<std::size_t tDim, AxisSide tSide>
  THES_ALWAYS_INLINE auto border_diagonal_summand(auto cell_value, auto tag) const {
    static_assert(non_zero_borders | thes::star::contains(tDim));
    return grex::constant<Real>(2 * sys_info_.axis_quotient(thes::index_tag<tDim>), tag) *
           std::get<tDim>(cell_value);
  }
  template<std::size_t tDim, AxisSide tSide>
  THES_ALWAYS_INLINE auto border_rhs_summand(auto cell_value, auto tag) const {
    static_assert(non_zero_borders | thes::star::contains(tDim));
    auto out = grex::constant<Real>(2 * sys_info_.axis_quotient(thes::index_tag<tDim>), tag) *
               std::get<tDim>(cell_value);
    if constexpr (tSide == AxisSide::START) {
      out *= grex::constant<Real>(sys_info_.solution_start(), tag);
    } else {
      out *= grex::constant<Real>(sys_info_.solution_end(), tag);
    }
    return out;
  }

  template<std::size_t tDim, AxisSide tSide>
  THES_ALWAYS_INLINE auto connection_value(auto from_info, auto to_info, auto tag) const {
    return diff_(std::get<tDim>(from_info), std::get<tDim>(to_info), tag) *
           grex::constant(sys_info_.axis_quotient(thes::index_tag<tDim>), tag);
  }

  template<std::size_t tDim, AxisSide tSide>
  THES_ALWAYS_INLINE auto off_diagonal(auto connection_value, auto /*tag*/) const {
    return -connection_value;
  }
  template<std::size_t tDim, AxisSide tSide>
  THES_ALWAYS_INLINE auto diagonal_summand(auto connection_value, auto /*tag*/) const {
    return connection_value;
  }

  THES_ALWAYS_INLINE auto diagonal(auto diagonal_sum, auto tag) const {
    if constexpr (zero_to_one) {
      return grex::select(diagonal_sum == grex::constant<Real>(0, tag),
                          grex::constant<Real>(1, tag), diagonal_sum, tag);
    } else {
      return diagonal_sum;
    }
  }

  const SystemInfo& info() const {
    return sys_info_;
  }

  const Storage& storage() const {
    return storage_;
  }
  Iter begin() const {
    return Iter{
      storage_ |
        thes::star::transform([this](const AxisStorage& s) { return s.begin() + data_offset_; }) |
        thes::star::to_array,
      0,
    };
  }

  Size size() const {
    return sys_info_.total_size();
  }

  const DistributedInfoStorage& distributed_info_storage() const {
    return dist_info_;
  }
  const DistributedInfoStorage& distributed_info() const
  requires(!is_shared)
  {
    return dist_info_;
  }

  [[nodiscard]] std::size_t data_size() const {
    assert(storage_ | thes::star::transform([this](const AxisStorage& s) { return s.size(); }) |
           thes::star::has_unique_value);
    return *(storage_ | thes::star::transform([this](const AxisStorage& s) { return s.size(); }) |
             thes::star::unique_value);
  }

private:
  RawSystemInfo sys_info_;
  Storage storage_;
  [[no_unique_address]] DiffusionComputer diff_;
  [[no_unique_address]] DistributedInfoStorage dist_info_;
  [[no_unique_address]] DataOffsetStorage data_offset_{};
};
} // namespace lineal

#endif // INCLUDE_LINEAL_TENSOR_DYNAMIC_STENCIL_VALUATOR_STATIONARY_DIFFUSION_VALUES_HPP
