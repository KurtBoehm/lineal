// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef TEST_MATERIALS_HPP
#define TEST_MATERIALS_HPP

#include <array>
#include <cstddef>
#include <functional>
#include <random>
#include <ranges>
#include <type_traits>
#include <utility>

#include "pcg_random.hpp"
#include "thesauros/algorithms.hpp"
#include "thesauros/format.hpp"
#include "thesauros/io.hpp"
#include "thesauros/macropolis.hpp"
#include "thesauros/math.hpp"
#include "thesauros/static-ranges.hpp"
#include "thesauros/types.hpp"
#include "thesauros/utility.hpp"

#include "lineal/component-wise.hpp"
#include "lineal/vectorization.hpp"

namespace lineal::materials {
namespace detail {
template<typename TDefs, typename TR1, typename TR2>
static auto from_centre(TR1&& r1, TR2&& r2) {
  using Real = TDefs::Real;
  return thes::star::transform(
    []<typename TSize>(TSize arg1, TSize arg2) {
      return thes::pow<2>(Real(arg1) / Real(arg2) - Real{0.5});
    },
    std::forward<TR1>(r1), std::forward<TR2>(r2));
}

template<typename TDefs, typename TR1, typename TR2>
static auto from_centre_sum(TR1&& r1, TR2&& r2) {
  using Real = TDefs::Real;
  return from_centre<TDefs>(std::forward<TR1>(r1), std::forward<TR2>(r2)) |
         thes::star::left_reduce(std::plus<>{}, Real{0});
}

template<typename TSize>
struct MaterialSinkConf {
  using Value = void;
  using Size = TSize;

  static constexpr bool supports_const_access = false;
  static constexpr bool supports_mutable_access = true;
};

template<typename TDefs>
using CellInfo = std::decay_t<typename TDefs::MaterialIndices>::Value;

template<typename TMatIdxs, typename TSize, typename TOp>
struct MaterialSink : public facades::SharedNullaryCwOp<MaterialSink<TMatIdxs, TSize, TOp>,
                                                        MaterialSinkConf<TSize>> {
  using Facade = facades::SharedNullaryCwOp<MaterialSink, MaterialSinkConf<TSize>>;
  using MatIdxs = std::decay_t<TMatIdxs>;
  using Size = TSize;

  MaterialSink(TMatIdxs&& mat, TOp op)
      : Facade(*thes::safe_cast<Size>(mat.size())), mat_(std::forward<TMatIdxs>(mat)),
        op_(std::move(op)) {}
  template<typename TDefs>
  MaterialSink(Size size, TOp op, thes::TypeTag<TDefs> /*tag*/)
      : Facade(size), mat_(size), op_(std::move(op)) {}

  THES_ALWAYS_INLINE void compute_impl(grex::ScalarTag /*tag*/, Size index) {
    mat_[index] = op_(index);
  }

  MaterialSink<TMatIdxs&, TSize, TOp> thread_instance(const auto& /*info*/,
                                                      grex::ScalarTag /*tag*/) {
    return {mat_, op_};
  }

  [[nodiscard]] Size size() const {
    return *thes::safe_cast<Size>(mat_.size());
  }

  TMatIdxs&& material_indices() && {
    return std::move(mat_);
  }

private:
  TMatIdxs mat_;
  TOp op_;
};
template<typename TDefs, typename TOp>
MaterialSink(typename TDefs::Size, TOp, thes::TypeTag<TDefs>)
  -> MaterialSink<std::decay_t<typename TDefs::MaterialIndices>, typename TDefs::Size, TOp>;
} // namespace detail

template<typename TDefs>
inline auto random(const auto& /*info*/, const auto range, const auto& expo) {
  using Size = TDefs::Size;
  using CellInfo = detail::CellInfo<TDefs>;
  using Dist = std::uniform_int_distribution<CellInfo>;
  static constexpr CellInfo max_material_num = TDefs::max_material_num;

  detail::MaterialSink sink{
    range.size(),
    [rng = pcg64{}, dist = Dist{0, max_material_num}](const Size /*index*/) mutable {
      return dist(rng);
    },
    thes::type_tag<TDefs>,
  };
  expo.execute(sink);
  return std::move(sink).material_indices();
}

template<typename TDefs>
inline auto flow_cylinder(const auto& info, const auto range, typename TDefs::Real radius,
                          const auto& expo) {
  using Size = TDefs::Size;
  using CellInfo = detail::CellInfo<TDefs>;
  constexpr auto flow_axis = TDefs::flow_axis;

  const auto goff = range.begin_value();
  detail::MaterialSink sink{
    range.size(),
    [&](const Size index) -> CellInfo {
      const auto pos = info.index_to_pos(index + goff);
      const auto value =
        detail::from_centre_sum<TDefs>(pos | thes::star::all_except_idxs<flow_axis>,
                                       info.sizes() | thes::star::all_except_idxs<flow_axis>);
      return (value <= radius) ? 3 : 0;
    },
    thes::type_tag<TDefs>,
  };
  expo.execute(sink);
  return std::move(sink).material_indices();
}

template<typename TDefs>
inline auto flow_cylinder_gap(const auto& info, const auto range, typename TDefs::Real radius,
                              const auto& expo) {
  using Size = TDefs::Size;
  using CellInfo = detail::CellInfo<TDefs>;
  constexpr std::size_t flow_axis = TDefs::flow_axis;

  const auto goff = range.begin_value();
  const Size mid = info.axis_size(thes::index_tag<flow_axis>) / 2;

  detail::MaterialSink sink{
    range.size(),
    [&](const Size index) -> CellInfo {
      const auto pos = info.index_to_pos(index + goff);
      const auto value =
        detail::from_centre_sum<TDefs>(pos | thes::star::all_except_idxs<flow_axis>,
                                       info.sizes() | thes::star::all_except_idxs<flow_axis>);
      return (value <= radius && std::get<flow_axis>(pos) != mid) ? 3 : 0;
    },
    thes::type_tag<TDefs>,
  };
  expo.execute(sink);
  return std::move(sink).material_indices();
}

template<typename TDefs>
inline auto flow_cylinder_layer(const auto& info, const auto range, typename TDefs::Real radius,
                                typename TDefs::Real layer_depth, const auto& expo) {
  using Real = TDefs::Real;
  using Size = TDefs::Size;
  using CellInfo = detail::CellInfo<TDefs>;
  constexpr auto flow_axis = TDefs::flow_axis;

  const auto goff = range.begin_value();
  const Real half_layer_depth = layer_depth / 2;

  detail::MaterialSink sink{
    range.size(),
    [&](const Size index) -> CellInfo {
      const auto pos = info.index_to_pos(index + goff);
      const auto values = detail::from_centre<TDefs>(pos, info.sizes());
      const auto base = thes::star::all_except_idxs<flow_axis>(values) |
                        thes::star::left_reduce(std::plus<>{}, Real{0});
      return (base <= radius)
               ? ((thes::star::get_at<flow_axis>(values) >= half_layer_depth) ? 3 : 1)
               : 0;
    },
    thes::type_tag<TDefs>,
  };
  expo.execute(sink);
  return std::move(sink).material_indices();
}

template<typename TDefs>
inline auto centred_sphere(const auto& info, const auto range, typename TDefs::Real radius,
                           const auto& expo) {
  using Size = TDefs::Size;
  using CellInfo = detail::CellInfo<TDefs>;

  const auto goff = range.begin_value();
  detail::MaterialSink sink{
    range.size(),
    [&](const Size index) {
      const auto pos = info.index_to_pos(index + goff);
      const auto value = detail::from_centre_sum<TDefs>(pos, info.sizes());
      return CellInfo{value <= radius};
    },
    thes::type_tag<TDefs>,
  };
  expo.execute(sink);
  return std::move(sink).material_indices();
}

template<typename TDefs>
inline auto diagonal_lines(const auto& info, const auto range, const auto& expo) {
  using Size = TDefs::Size;
  using CellInfo = detail::CellInfo<TDefs>;

  detail::MaterialSink sink{
    range.size(),
    [&](const Size index) {
      const auto pos = info.index_to_pos(range[index]);
      const auto base = thes::star::all_except_idxs<TDefs::flow_axis>(pos) |
                        thes::star::left_reduce(std::plus<>{}, Size{0});
      return *thes::safe_cast<CellInfo>(base % TDefs::max_material_num);
    },
    thes::type_tag<TDefs>,
  };
  expo.execute(sink);
  return std::move(sink).material_indices();
}

template<typename TDefs>
inline auto diagonal_lines_eff_diff(const auto& info) {
  using Size = TDefs::Size;
  using Real = TDefs::Real;

  const auto& mat_coeffs = info.material_coefficients();
  const auto sizes = info.sizes() | thes::star::all_except_idxs<TDefs::flow_axis>;

  Real sum = 0;
  thes::multidim_for_each_size(sizes, [&](auto... coords) {
    const auto base = std::array{coords...} | thes::star::left_reduce(std::plus{}, Size{0});
    sum += mat_coeffs[base % TDefs::max_material_num];
  });
  return sum / Real(sizes | thes::star::left_reduce(std::multiplies{}));
}

template<typename TDefs, typename TSize, std::size_t tDims>
inline auto expanded_repeated(auto make_base, const std::array<TSize, tDims>& ppos,
                              const thes::SubMultiSize<TSize, tDims>& dims_off,
                              const thes::SubMultiSize<TSize, tDims>& dims_exp,
                              const thes::MultiSize<TSize, tDims>& world_dims, const auto& expo) {
  using Pos = std::array<TSize, tDims>;
  using Size = TDefs::Size;

  const auto reflect = thes::star::static_apply<tDims>(
    [&]<std::size_t... tI>() { return std::array<bool, tDims>{(get<tI>(ppos) % 2 == 1)...}; });
  const auto base_materials =
    make_base(std::views::transform(thes::range(dims_off.total_size()), [&](auto i) {
      return world_dims.pos_to_index(dims_off.local_index_to_global_pos(i));
    }));

  detail::MaterialSink sink{
    dims_exp.total_size(),
    [&](const Size i) {
      const Pos gpos = dims_exp.local_index_to_global_pos(i);
      const Pos rpos = dims_off.reflect(dims_off.reflect_into(gpos), reflect);
      return base_materials[dims_off.global_pos_to_local_index(rpos)];
    },
    thes::type_tag<TDefs>,
  };
  expo.execute(sink);
  return std::move(sink).material_indices();
}

template<typename TMultiSizes, typename TMaterials, bool tHorizontal>
struct MaterialIndexPrinter {
  TMultiSizes multisizes;
  TMaterials materials;
};
template<bool tHorizontal = true, typename TMultiSizes, typename TMaterials>
inline MaterialIndexPrinter<TMultiSizes, TMaterials, tHorizontal>
index_printer(TMultiSizes&& multisizes, TMaterials&& materials) {
  return {
    .multisizes = std::forward<TMultiSizes>(multisizes),
    .materials = std::forward<TMaterials>(materials),
  };
}
} // namespace lineal::materials

template<typename TMultiSizes, typename TMaterials, bool tHorizontal>
struct fmt::formatter<lineal::materials::MaterialIndexPrinter<TMultiSizes, TMaterials, tHorizontal>>
    : public fmt::nested_formatter<typename std::decay_t<TMaterials>::Value> {
  auto format(
    const lineal::materials::MaterialIndexPrinter<TMultiSizes, TMaterials, tHorizontal>& printer,
    fmt::format_context& ctx) const {
    return this->write_padded(ctx, [&](auto it) {
      using Size = std::decay_t<TMultiSizes>::Size;
      constexpr std::size_t dimensions = std::decay_t<TMultiSizes>::dimension_num;
      constexpr auto dim1 = thes::index_tag<dimensions - 1>;
      constexpr auto dim2 = thes::index_tag<dimensions - 2>;
      const auto& ms = printer.multisizes;

      if constexpr (tHorizontal) {
        for (thes::Delimiter d{"\n"}; const auto i : thes::range(ms.axis_size(dim2))) {
          fmt::format_to(it, "{}", d);
          for (thes::Delimiter dd{" "};
               const auto j : thes::range(Size{}, ms.total_size(), ms.from_size(dim2))) {
            fmt::format_to(it, "{}", dd);
            for (const auto k : thes::range(ms.axis_size(dim1))) {
              const auto v = printer.materials[j + i * ms.axis_size(dim1) + k];
              fmt::format_to(it, thes::rainbow_fg(v), "{}", v);
            }
          }
        }
      } else {
        for (thes::Delimiter d{"\n-\n"};
             const auto& chunk : std::views::chunk(printer.materials, ms.from_size(dim2))) {
          fmt::format_to(it, "{}", d);
          for (thes::Delimiter dd{"\n"};
               const auto& row : std::views::chunk(chunk, ms.axis_size(dim1))) {
            fmt::format_to(it, "{}", dd);
            for (const auto& v : row) {
              fmt::format_to(it, thes::rainbow_fg(v), "{}", v);
            }
          }
        }
      }
      return it;
    });
  }
};

#endif // TEST_MATERIALS_HPP
