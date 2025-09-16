// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_TENSOR_DYNAMIC_STENCIL_VALUATOR_STATIONARY_DIFFUSION_LOOKUP_HPP
#define INCLUDE_LINEAL_TENSOR_DYNAMIC_STENCIL_VALUATOR_STATIONARY_DIFFUSION_LOOKUP_HPP

#include <array>
#include <cassert>
#include <cstddef>
#include <span>
#include <type_traits>
#include <utility>

#include "thesauros/macropolis/inlining.hpp"
#include "thesauros/static-ranges.hpp"
#include "thesauros/types.hpp"

#include "lineal/base.hpp"
#include "lineal/parallel.hpp"
#include "lineal/tensor/dynamic/stencil/def.hpp"
#include "lineal/tensor/dynamic/stencil/valuator/base.hpp"
#include "lineal/vectorization.hpp"

namespace lineal {
template<typename TDefs, AnyZeroHandlingTag TZeroHandling, OptDistributedInfo TDistInfo>
struct StationaryDiffusionLookupValuator : public ValuatorBase, public TDefs {
  using Real = TDefs::Real;
  using Size = TDefs::Size;
  using RealSize = grex::FloatSize<Real>;

  using RawSystemInfo = TDefs::SystemInfo;
  using SystemInfo = std::decay_t<RawSystemInfo>;
  using RawMaterialIndices = TDefs::MaterialIndices;
  using MaterialIndices = std::decay_t<RawMaterialIndices>;
  using CellInfo = MaterialIndices::Value;
  using Iter = ValuatorPtrIter<CellInfo>;
  using LookupWright = TDefs::LookupWright;

  using RawDistributedInfo = TDistInfo;
  using DistributedInfo = std::decay_t<TDistInfo>;
  using DistributedInfoStorage = thes::VoidStorage<TDistInfo>;
  static constexpr bool is_shared = std::is_void_v<TDistInfo>;

  static constexpr bool zero_to_one = TZeroHandling::value;
  static constexpr std::size_t dimension_num = TDefs::dimension_num;
  static constexpr std::size_t max_material_num = TDefs::max_material_num;
  static constexpr std::size_t flow_axis = TDefs::flow_axis;

  static constexpr std::array non_zero_borders{flow_axis};
  static constexpr bool is_symmetric = LookupWright::is_symmetric;

  using ConnectionLookup =
    std::array<std::array<Real, max_material_num * max_material_num>, dimension_num>;
  using DiffusionFactorLookup = std::array<Real, max_material_num * max_material_num>;
  using FilledLookup = std::array<bool, max_material_num>;
  using FilledNumLookup = std::array<Size, max_material_num>;
  using DataOffsetStorage = std::conditional_t<is_shared, thes::ValueTag<Size, 0>, Size>;

  explicit StationaryDiffusionLookupValuator(RawSystemInfo&& system_info,
                                             RawMaterialIndices&& material,
                                             thes::VoidStorageRvalRef<TDistInfo> dist_info = {},
                                             const LookupWright& maker = LookupWright{})
      : sys_info_(std::forward<RawSystemInfo>(system_info)),
        material_(std::forward<RawMaterialIndices>(material)),
        connection_lookup_(maker.make_connection_lookup(sys_info_)),
        diffusion_factor_lookup_(maker.make_diffusion_factor_lookup(sys_info_)),
        filled_lookup_(maker.make_filled_lookup(sys_info_)),
        filled_num_lookup_(filled_lookup_ | thes::star::transform([](bool b) { return Size{b}; }) |
                           thes::star::to_array),
        dist_info_(std::forward<DistributedInfoStorage>(dist_info)) {
    if constexpr (is_shared) {
      assert(sys_info_.total_size() == material_.size());
    } else {
      const auto range = material_range(sys_info_, dist_info_);
      assert(range.size() == material_.size());
      data_offset_ = dist_info_.local_begin(global_index_tag) - range.begin_value();
    }
  }

  THES_ALWAYS_INLINE auto diffusion_coeff(Size from, Size to, grex::AnyTag auto tag) const {
    auto* ptr = material_.data() + data_offset_;
    const auto mat_from = grex::convert_safe<RealSize>(grex::load(ptr + from, tag));
    const auto mat_to = grex::convert_safe<RealSize>(grex::load(ptr + to, tag));
    return grex::gather(std::span{diffusion_factor_lookup_}, mat_from * max_material_num + mat_to,
                        tag);
  }

  THES_ALWAYS_INLINE bool is_filled(Size idx, grex::ScalarTag /*tag*/) const {
    auto* ptr = material_.data() + data_offset_;
    return filled_lookup_[ptr[idx]];
  }
  THES_ALWAYS_INLINE auto is_filled_num(Size idx, grex::AnyTag auto tag) const {
    auto* ptr = material_.data() + data_offset_;
    const auto mat = grex::load(ptr + idx, tag);
    return grex::gather(std::span{filled_num_lookup_}, mat, tag);
  }

  THES_ALWAYS_INLINE auto cell_value(auto idx, grex::AnyTag auto tag) const {
    return grex::gather(std::span{coeffs()}, idx, tag);
  }

  THES_ALWAYS_INLINE auto diagonal_base(auto /*cell_value*/, auto tag) const {
    return grex::zeros<Real>(tag);
  }

  template<std::size_t tDim, AxisSide tSide>
  THES_ALWAYS_INLINE auto border_diagonal_summand(auto cell_value, auto tag) const {
    static_assert(non_zero_borders | thes::star::contains(tDim));
    return grex::broadcast<Real>(2 * sys_info_.axis_quotient(thes::index_tag<tDim>), tag) *
           cell_value;
  }
  template<std::size_t tDim, AxisSide tSide>
  THES_ALWAYS_INLINE auto border_rhs_summand(auto cell_value, auto tag) const {
    static_assert(non_zero_borders | thes::star::contains(tDim));
    auto out =
      grex::broadcast<Real>(2 * sys_info_.axis_quotient(thes::index_tag<tDim>), tag) * cell_value;
    if constexpr (tSide == AxisSide::START) {
      out *= grex::broadcast<Real>(sys_info_.solution_start(), tag);
    } else {
      out *= grex::broadcast<Real>(sys_info_.solution_end(), tag);
    }
    return out;
  }

  template<std::size_t tDim, AxisSide tSide>
  THES_ALWAYS_INLINE auto connection_value(auto from_cell_info, auto to_cell_info, auto tag) const {
    auto from_extended = grex::convert_safe<RealSize>(from_cell_info);
    auto to_extended = grex::convert_safe<RealSize>(to_cell_info);
    const auto size_tag = tag.cast(grex::type_tag<RealSize>);
    return grex::gather(
      std::span{std::get<tDim>(connection_lookup_)},
      from_extended * grex::broadcast<RealSize>(max_material_num, size_tag) + to_extended, tag);
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
      return grex::blend(diagonal_sum != grex::zeros<Real>(tag), grex::broadcast<Real>(1, tag),
                         diagonal_sum);
    } else {
      return diagonal_sum;
    }
  }

  const SystemInfo& info() const {
    return sys_info_;
  }

  const MaterialIndices& cell_infos() const {
    return material_;
  }

  Iter begin() const {
    return Iter{material_.data() + data_offset_};
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

  Size data_offset() const {
    return data_offset_;
  }
  [[nodiscard]] std::size_t data_size() const {
    return material_.size();
  }

private:
  decltype(auto) coeffs() const {
    return sys_info_.material_coefficients();
  }

  RawSystemInfo sys_info_;
  RawMaterialIndices material_;
  ConnectionLookup connection_lookup_;
  DiffusionFactorLookup diffusion_factor_lookup_;
  FilledLookup filled_lookup_;
  FilledNumLookup filled_num_lookup_;
  [[no_unique_address]] DistributedInfoStorage dist_info_;
  [[no_unique_address]] DataOffsetStorage data_offset_{};
};
} // namespace lineal

#endif // INCLUDE_LINEAL_TENSOR_DYNAMIC_STENCIL_VALUATOR_STATIONARY_DIFFUSION_LOOKUP_HPP
