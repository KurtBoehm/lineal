// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_TENSOR_DYNAMIC_STENCIL_MATRIX_DIFFUSION_HPP
#define INCLUDE_LINEAL_TENSOR_DYNAMIC_STENCIL_MATRIX_DIFFUSION_HPP

#include <cstddef>

#include "thesauros/containers.hpp"

#include "lineal/discretization/box-equidistant.hpp"
#include "lineal/tensor/dynamic/stencil/valuator/lookup-wright.hpp"

namespace lineal {
template<typename TDefs>
struct SymmetricDiffusionInfo
    : public BoxInfo<typename TDefs::Real, typename TDefs::Size, TDefs::dimension_num> {
  using Defs = TDefs;
  using Real = Defs::Real;
  using Size = Defs::Size;
  static constexpr std::size_t dimension_num = Defs::dimension_num;
  static constexpr std::size_t max_material_num = Defs::max_material_num;

  using Parent = BoxInfo<Real, Size, dimension_num>;
  using AxisArray = Parent::SizeArray;
  using RealArray = Parent::RealArray;

  using LookupWright = SymmetricDiffusionLookupWright;

  constexpr SymmetricDiffusionInfo(AxisArray axis_cells, RealArray axis_lengths,
                                   Real solution_start, Real solution_end)
      : Parent(axis_cells, axis_lengths), solution_start_(solution_start),
        solution_end_(solution_end) {}

  [[nodiscard]] constexpr Real solution_start() const {
    return solution_start_;
  }
  [[nodiscard]] constexpr Real solution_end() const {
    return solution_end_;
  }

  bool operator==(const SymmetricDiffusionInfo&) const = default;

private:
  Real solution_start_;
  Real solution_end_;
};

template<typename TDefs>
struct LookupSymmetricDiffusionInfo
    : public BoxInfo<typename TDefs::Real, typename TDefs::Size, TDefs::dimension_num> {
  using Defs = TDefs;
  using Real = Defs::Real;
  using Size = Defs::Size;
  static constexpr std::size_t dimension_num = Defs::dimension_num;
  static constexpr std::size_t max_material_num = Defs::max_material_num;

  using Parent = BoxInfo<Real, Size, dimension_num>;
  using AxisArray = Parent::SizeArray;
  using RealArray = Parent::RealArray;

  using MaterialCoeffs = thes::LimitedArray<Real, max_material_num>;
  using NoneMaterialMap = thes::LimitedArray<bool, max_material_num>;
  using LookupWright = SymmetricDiffusionLookupWright;

  constexpr LookupSymmetricDiffusionInfo(AxisArray axis_cells, RealArray axis_lengths,
                                         MaterialCoeffs material_coefficients,
                                         NoneMaterialMap none_mats, Real solution_start,
                                         Real solution_end)
      : Parent(axis_cells, axis_lengths), material_coeffs_(material_coefficients),
        none_mats_(none_mats), solution_start_(solution_start), solution_end_(solution_end) {}

  [[nodiscard]] constexpr const MaterialCoeffs& material_coefficients() const {
    return material_coeffs_;
  }
  [[nodiscard]] constexpr const NoneMaterialMap& none_materials() const {
    return none_mats_;
  }
  [[nodiscard]] constexpr Real solution_start() const {
    return solution_start_;
  }
  [[nodiscard]] constexpr Real solution_end() const {
    return solution_end_;
  }

  bool operator==(const LookupSymmetricDiffusionInfo&) const = default;

private:
  MaterialCoeffs material_coeffs_;
  NoneMaterialMap none_mats_;
  Real solution_start_;
  Real solution_end_;
};

template<typename TDefs>
struct LookupHenryDiffusionInfo
    : public BoxInfo<typename TDefs::Real, typename TDefs::Size, TDefs::dimension_num> {
  using Defs = TDefs;
  using Real = Defs::Real;
  using Size = Defs::Size;
  static constexpr std::size_t dimension_num = Defs::dimension_num;
  static constexpr std::size_t max_material_num = Defs::max_material_num;

  using Parent = BoxInfo<Real, Size, dimension_num>;
  using AxisArray = Parent::SizeArray;
  using RealArray = Parent::RealArray;

  using MaterialCoeffs = thes::LimitedArray<Real, max_material_num>;
  using MaterialGaseous = thes::LimitedArray<bool, max_material_num>;
  using NoneMaterialMap = thes::LimitedArray<bool, max_material_num>;
  using LookupWright = HenryDiffusionLookupWright;

  constexpr LookupHenryDiffusionInfo(AxisArray axis_cells, RealArray axis_lengths,
                                     MaterialCoeffs material_coefficients,
                                     MaterialGaseous material_gaseous, NoneMaterialMap none_mats,
                                     Real henry_constant, Real solution_start, Real solution_end)
      : Parent(axis_cells, axis_lengths), material_coefficients_(material_coefficients),
        material_gaseous_(material_gaseous), none_mats_(none_mats), henry_constant_(henry_constant),
        solution_start_(solution_start), solution_end_(solution_end) {}

  [[nodiscard]] constexpr const MaterialCoeffs& material_coefficients() const {
    return material_coefficients_;
  }
  [[nodiscard]] constexpr const MaterialGaseous& material_gaseous() const {
    return material_gaseous_;
  }
  [[nodiscard]] constexpr const NoneMaterialMap& none_materials() const {
    return none_mats_;
  }

  [[nodiscard]] constexpr Real henry_constant() const {
    return henry_constant_;
  }
  [[nodiscard]] constexpr Real solution_start() const {
    return solution_start_;
  }
  [[nodiscard]] constexpr Real solution_end() const {
    return solution_end_;
  }

private:
  MaterialCoeffs material_coefficients_;
  MaterialGaseous material_gaseous_;
  NoneMaterialMap none_mats_;
  Real henry_constant_;
  Real solution_start_;
  Real solution_end_;
};
} // namespace lineal

#endif // INCLUDE_LINEAL_TENSOR_DYNAMIC_STENCIL_MATRIX_DIFFUSION_HPP
