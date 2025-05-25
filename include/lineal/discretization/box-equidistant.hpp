// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_DISCRETIZATION_BOX_EQUIDISTANT_HPP
#define INCLUDE_LINEAL_DISCRETIZATION_BOX_EQUIDISTANT_HPP

#include <array>
#include <cassert>
#include <cstddef>
#include <functional>

#include "thesauros/static-ranges.hpp"
#include "thesauros/types.hpp"
#include "thesauros/utility.hpp"

#include "lineal/vectorization.hpp"

namespace lineal {
// Information on the number of cells in each dimension
template<typename TSize, std::size_t tDims>
struct CellInfo : public thes::MultiSize<TSize, tDims> {
  using Base = thes::MultiSize<TSize, tDims>;
  using Base::Base;
  using Base::dimension_num;
  using Base::index_to_axis_index;

  template<std::size_t tDim, typename TIndex>
  [[nodiscard]] auto axis_index(TIndex dim_index, auto tag, thes::IndexTag<tDim> dim = {}) const {
    if constexpr (tDim + 1 < dimension_num) {
      return grex::constant<TIndex>(dim_index, tag);
    } else {
      assert(tag.part() <= this->axis_size(dim));
      return grex::index_from(dim_index, tag);
    }
  }

  template<typename TIndex, std::size_t tDim, typename TVecTag>
  [[nodiscard]] constexpr grex::Vector<TIndex, TVecTag::size>
  index_to_axis_index(TSize index, TVecTag tag, thes::TypeTag<TIndex> /*tag*/ = {},
                      thes::IndexTag<tDim> dim = {}) const {
    const TIndex div = (index % this->from_size_div(dim)) / this->after_size_div(dim);
    return axis_index(div, tag, dim);
  }
  template<std::size_t tDim, typename TVecTag>
  [[nodiscard]] constexpr grex::Vector<TSize, TVecTag::size>
  index_to_axis_index(TSize index, TVecTag tag, thes::IndexTag<tDim> dim = {}) const {
    return index_to_axis_index(index, tag, thes::type_tag<TSize>, dim);
  }

  bool operator==(const CellInfo&) const = default;
};

template<typename TReal, typename TSize, std::size_t tDimNum>
struct BoxInfo : public CellInfo<TSize, tDimNum> {
  using Real = TReal;
  using Size = TSize;
  static constexpr std::size_t dimension_num = tDimNum;

  using Parent = CellInfo<TSize, tDimNum>;
  using RealArray = std::array<Real, dimension_num>;
  using SizeArray = Parent::AxisSize;

  explicit constexpr BoxInfo(SizeArray cell_nums, RealArray axis_lengths)
      : Parent(cell_nums), axis_lengths_(axis_lengths),
        axis_quotients_(
          thes::star::index_transform<dimension_num>(
            [axis_distances =
               thes::star::transform([](Real arg1, Size arg2) { return arg1 / Real(arg2); },
                                     axis_lengths, cell_nums) |
               thes::star::to_array](auto i) {
              const Real axis_area = thes::star::all_except_idxs<i>(axis_distances) |
                                     thes::star::left_reduce(std::multiplies{}, Real{1});
              return axis_area / std::get<i>(axis_distances);
            }) |
          thes::star::to_array) {}

  [[nodiscard]] constexpr RealArray axis_lengths() const {
    return axis_lengths_;
  }
  template<std::size_t tDim>
  [[nodiscard]] constexpr Real axis_length(thes::IndexTag<tDim> /*tag*/) const {
    return std::get<tDim>(axis_lengths_);
  }
  [[nodiscard]] constexpr Real axis_length(std::size_t axis) const {
    return axis_lengths_[axis];
  }
  [[nodiscard]] constexpr RealArray axis_quotients() const {
    return axis_quotients_;
  }
  template<std::size_t tDim>
  [[nodiscard]] constexpr Real axis_quotient(thes::IndexTag<tDim> /*tag*/) const {
    return std::get<tDim>(axis_quotients_);
  }
  [[nodiscard]] constexpr Real axis_quotient(std::size_t axis) const {
    return axis_quotients_[axis];
  }
  template<std::size_t tDim>
  [[nodiscard]] constexpr Real other_length(thes::IndexTag<tDim> /*tag*/) const {
    return axis_lengths_ | thes::star::all_except_idxs<tDim> |
           thes::star::left_reduce(std::multiplies{}, Real{1});
  }

private:
  RealArray axis_lengths_;
  RealArray axis_quotients_;
};
} // namespace lineal

#endif // INCLUDE_LINEAL_DISCRETIZATION_BOX_EQUIDISTANT_HPP
