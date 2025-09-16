// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_TENSOR_DYNAMIC_STENCIL_VALUATOR_LOOKUP_WRIGHT_HPP
#define INCLUDE_LINEAL_TENSOR_DYNAMIC_STENCIL_VALUATOR_LOOKUP_WRIGHT_HPP

#include <cstddef>

#include "thesauros/static-ranges.hpp"
#include "thesauros/types.hpp"

#include "lineal/math/harmonic-mean.hpp"

namespace lineal {
struct SymmetricDiffusionLookupProvider {
  static constexpr bool is_symmetric = true;

  template<typename TSysInfo>
  static constexpr auto diffusion_factor(const TSysInfo& sys_info,
                                         thes::AnyIndexTag auto flat_idx) {
    constexpr std::size_t max_material_num = TSysInfo::max_material_num;

    const auto& cell_lookup = sys_info.material_coefficients();
    const auto diff_from = cell_lookup[flat_idx / max_material_num];
    const auto diff_to = cell_lookup[flat_idx % max_material_num];

    return harmonic_mean(diff_from, diff_to);
  }

  static constexpr auto diffusion_factor_computer() {
    return [](auto diff_from, auto diff_to) { return harmonic_mean(diff_from, diff_to); };
  }
};

struct HenryDiffusionLookupProvider {
  static constexpr bool is_symmetric = true;

  template<typename TSysInfo>
  static constexpr auto diffusion_factor(const TSysInfo& sys_info,
                                         thes::AnyIndexTag auto flat_idx) {
    using Real = TSysInfo::Real;
    constexpr std::size_t max_material_num = TSysInfo::max_material_num;

    constexpr auto from_idx = flat_idx / max_material_num;
    constexpr auto to_idx = flat_idx % max_material_num;

    const auto& cell_lookup = sys_info.material_coefficients();
    const auto diff_from = cell_lookup[from_idx];
    const auto diff_to = cell_lookup[to_idx];

    const auto gaseous_lookup = sys_info.material_gaseous();
    const auto henry = sys_info.henry_constant();
    const auto factor_from = gaseous_lookup[from_idx] ? henry : Real{1};
    const auto factor_to = gaseous_lookup[to_idx] ? henry : Real{1};

    return harmonic_mean(diff_from * factor_to, diff_to * factor_from);
  }
};

template<typename TProvider>
struct DefaultLookupWright {
  static constexpr bool is_symmetric = TProvider::is_symmetric;

  static constexpr auto diffusion_factor_computer()
  requires(requires { TProvider::diffusion_factor_computer(); })
  {
    return TProvider::diffusion_factor_computer();
  }

  template<typename TSysInfo>
  static constexpr auto make_connection_lookup(const TSysInfo& sys_info) {
    constexpr std::size_t dimension_num = TSysInfo::dimension_num;
    constexpr std::size_t max_material_num = TSysInfo::max_material_num;

    return thes::star::index_transform<dimension_num>([&](auto dimension) {
             return thes::star::index_transform<max_material_num * max_material_num>(
                      [&](auto flat_idx) {
                        return TProvider::diffusion_factor(sys_info, flat_idx) *
                               sys_info.axis_quotient(dimension);
                      }) |
                    thes::star::to_array;
           }) |
           thes::star::to_array;
  }

  template<typename TSysInfo>
  static constexpr auto make_diffusion_factor_lookup(const TSysInfo& sys_info) {
    constexpr std::size_t max_material_num = TSysInfo::max_material_num;

    return thes::star::index_transform<max_material_num * max_material_num>(
             [&](auto flat_idx) { return TProvider::diffusion_factor(sys_info, flat_idx); }) |
           thes::star::to_array;
  }

  template<typename TSysInfo>
  static constexpr auto make_filled_lookup(const TSysInfo& sys_info) {
    constexpr std::size_t max_material_num = TSysInfo::max_material_num;

    decltype(auto) none_mats = sys_info.none_materials();
    return thes::star::index_transform<max_material_num>(
             [&](auto idx) { return !none_mats[idx]; }) |
           thes::star::to_array;
  }
};

using SymmetricDiffusionLookupWright = DefaultLookupWright<SymmetricDiffusionLookupProvider>;
using HenryDiffusionLookupWright = DefaultLookupWright<HenryDiffusionLookupProvider>;
} // namespace lineal

#endif // INCLUDE_LINEAL_TENSOR_DYNAMIC_STENCIL_VALUATOR_LOOKUP_WRIGHT_HPP
