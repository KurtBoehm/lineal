// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_TENSOR_DYNAMIC_STENCIL_MATRIX_ALGORITHM_HPP
#define INCLUDE_LINEAL_TENSOR_DYNAMIC_STENCIL_MATRIX_ALGORITHM_HPP

#include <cassert>
#include <cmath>
#include <functional>
#include <limits>
#include <optional>

#include "thesauros/functional.hpp"
#include "thesauros/macropolis.hpp"
#include "thesauros/static-ranges.hpp"
#include "thesauros/types.hpp"
#include "thesauros/utility.hpp"

#include "lineal/base.hpp"
#include "lineal/parallel.hpp"
#include "lineal/tensor/dynamic/stencil/def.hpp"
#include "lineal/vectorization.hpp"

namespace lineal::stencil {
template<typename TFilter, typename TValuator, typename TBeforeOp, typename TDiagOp,
         typename TAfterOp>
THES_ALWAYS_INLINE inline decltype(auto)
iterate(const auto idx, const auto pos, TFilter filter, const TValuator& valuator,
        TBeforeOp before_op, TDiagOp diagonal_op, TAfterOp after_op, UnvaluedTag /*is_valued*/,
        AnyOrderingTag auto /*is_ordered*/) {
  decltype(auto) sys_info = valuator.info();

  auto inner_lambda = [&](thes::AnyIndexTag auto j, thes::AnyBoolTag auto is_before)
                        THES_ALWAYS_INLINE -> decltype(auto) {
    const auto pos_j = std::get<j>(pos);
    const auto offset = sys_info.after_size(j);

    if constexpr (is_before) {
      using Ret = decltype(before_op(idx - offset));
      if (pos_j > 0 && filter(idx, offset, std::minus{})) [[likely]] {
        THES_APPLY_VALUED_RETURN(Ret, before_op(idx - offset));
      }
      THES_RETURN_EMPTY_OPTIONAL(Ret);
    } else {
      using Ret = decltype(after_op(idx + offset));
      if (pos_j + 1 < sys_info.axis_size(j) && filter(idx, offset, std::plus{})) [[likely]] {
        THES_APPLY_VALUED_RETURN(Ret, after_op(idx + offset));
      }
      THES_RETURN_EMPTY_OPTIONAL(Ret);
    }
  };
  auto outer_lambda = [&](thes::AnyBoolTag auto is_before) THES_ALWAYS_INLINE -> decltype(auto) {
    constexpr auto rng = [&]() THES_ALWAYS_INLINE {
      if constexpr (is_before) {
        return thes::star::iota<0, TValuator::dimension_num>;
      } else {
        return thes::star::iota<0, TValuator::dimension_num> | thes::star::reversed;
      }
    }();
    return rng | thes::star::transform([&](auto j) THES_ALWAYS_INLINE {
             return inner_lambda(j, is_before);
           }) |
           thes::star::first_value;
  };

  constexpr auto true_tag = thes::auto_tag<true>;
  constexpr auto false_tag = thes::auto_tag<false>;

  using Ret = thes::TypeSeq<decltype(outer_lambda(true_tag)), decltype(diagonal_op(idx)),
                            decltype(outer_lambda(false_tag))>::Unique;

  if constexpr (!thes::AnyNoOp<TBeforeOp>) {
    // before
    THES_APPLY_VALUED_RETURN(Ret, outer_lambda(true_tag));
  }
  if constexpr (!thes::AnyNoOp<TDiagOp>) {
    // diagonal
    THES_APPLY_VALUED_RETURN(Ret, diagonal_op(idx));
  }
  if constexpr (!thes::AnyNoOp<TAfterOp>) {
    // after
    THES_APPLY_VALUED_RETURN(Ret, outer_lambda(false_tag));
  }

  THES_RETURN_EMPTY_OPTIONAL(Ret);
}

template<typename TIdx, typename TFilter, typename TValuator, typename TBeforeOp, typename TDiagOp,
         typename TAfterOp>
THES_ALWAYS_INLINE inline decltype(auto)
iterate(const TIdx idx, const auto pos, TFilter filter, const TValuator& valuator,
        TBeforeOp before_op, TDiagOp diagonal_op, TAfterOp after_op, ValuedTag /*is_valued*/,
        AnyOrderingTag auto is_ordered) {
  using Real = TValuator::Real;
  using Size = TValuator::Size;
  using CellInfo = TValuator::CellInfo;

  decltype(auto) sys_info = valuator.info();
  const auto cell_infos = valuator.begin() + index_value(idx);
  const auto cell_info = *cell_infos;
  const auto cell_value = valuator.cell_value(cell_info, grex::scalar_tag);

  Real diagonal = valuator.diagonal_base(cell_value, grex::scalar_tag);
  if constexpr (!thes::AnyNoOp<TDiagOp>) {
    thes::star::iota<0, TValuator::dimension_num> |
      thes::star::only_range<TValuator::non_zero_borders> |
      thes::star::for_each([&](thes::AnyIndexTag auto j) THES_ALWAYS_INLINE {
        const Size pos_j = std::get<j>(pos);
        if (pos_j == 0) [[unlikely]] {
          diagonal += valuator.template border_diagonal_summand<j, AxisSide::START>(
            cell_value, grex::scalar_tag);
        }
        if (pos_j + 1 == sys_info.axis_size(j)) [[unlikely]] {
          diagonal += valuator.template border_diagonal_summand<j, AxisSide::END>(cell_value,
                                                                                  grex::scalar_tag);
        }
      });
  }

  auto inner_lambda = [&](thes::AnyIndexTag auto j, thes::AnyBoolTag auto is_before,
                          thes::AnyBoolTag auto call, thes::AnyBoolTag auto add) -> decltype(auto) {
    const Size pos_j = std::get<j>(pos);
    const Size offset = sys_info.after_size(j);

    if constexpr (is_before) {
      using Ret = decltype(before_op(std::declval<TIdx>(), std::declval<Real>()));

      if (pos_j > 0) [[likely]] {
        const TIdx prev = idx - offset;
        const CellInfo side_info = *(cell_infos - offset);
        const Real diff_coeff = valuator.template connection_value<j, AxisSide::START>(
          cell_info, side_info, grex::scalar_tag);
        if constexpr (call && !thes::AnyNoOp<TBeforeOp>) {
          if (filter(idx, offset, std::minus{})) {
            const Real off_diag =
              valuator.template off_diagonal<j, AxisSide::START>(diff_coeff, grex::scalar_tag);
            THES_APPLY_VALUED_RETURN(Ret, before_op(prev, off_diag));
          }
        }
        if constexpr (!thes::AnyNoOp<TDiagOp> && add) {
          diagonal +=
            valuator.template diagonal_summand<j, AxisSide::START>(diff_coeff, grex::scalar_tag);
        }
      }

      if constexpr (call) {
        THES_RETURN_EMPTY_OPTIONAL(Ret);
      }
    } else {
      using Ret = decltype(after_op(std::declval<TIdx>(), std::declval<Real>()));

      if (pos_j + 1 < sys_info.axis_size(j)) [[likely]] {
        const TIdx succ = idx + offset;
        const CellInfo side_info = *(cell_infos + offset);
        const Real diff_coeff = valuator.template connection_value<j, AxisSide::END>(
          cell_info, side_info, grex::scalar_tag);
        if constexpr (call && !thes::AnyNoOp<TAfterOp>) {
          if (filter(idx, offset, std::plus{})) {
            const Real off_diag =
              valuator.template off_diagonal<j, AxisSide::END>(diff_coeff, grex::scalar_tag);
            THES_APPLY_VALUED_RETURN(Ret, after_op(succ, off_diag));
          }
        }
        if constexpr (!thes::AnyNoOp<TDiagOp> && add) {
          diagonal +=
            valuator.template diagonal_summand<j, AxisSide::END>(diff_coeff, grex::scalar_tag);
        }
      }

      if constexpr (call) {
        THES_RETURN_EMPTY_OPTIONAL(Ret);
      }
    }
  };
  auto outer_lambda = [&](thes::AnyBoolTag auto is_before, thes::AnyBoolTag auto call,
                          thes::AnyBoolTag auto add) THES_ALWAYS_INLINE -> decltype(auto) {
    constexpr auto rng = [&]() THES_ALWAYS_INLINE {
      if constexpr (is_before) {
        return thes::star::iota<0, TValuator::dimension_num>;
      } else {
        return thes::star::iota<0, TValuator::dimension_num> | thes::star::reversed;
      }
    }();
    return rng | thes::star::transform([&](auto j) THES_ALWAYS_INLINE {
             return inner_lambda(j, is_before, call, add);
           }) |
           thes::star::first_value;
  };

  constexpr auto true_tag = thes::auto_tag<true>;
  constexpr auto false_tag = thes::auto_tag<false>;

  if constexpr (!thes::AnyNoOp<TDiagOp> && is_ordered) {
    using Ret = thes::TypeSeq<decltype(outer_lambda(true_tag, true_tag, true_tag)),
                              decltype(diagonal_op(idx, diagonal)),
                              decltype(outer_lambda(false_tag, true_tag, false_tag))>::Unique;

    // before
    THES_APPLY_VALUED_RETURN(Ret, outer_lambda(true_tag, true_tag, true_tag));
    // after to compute diagonal
    outer_lambda(false_tag, false_tag, true_tag);
    // diagonal
    diagonal = valuator.diagonal(diagonal, grex::scalar_tag);
    THES_APPLY_VALUED_RETURN(Ret, diagonal_op(idx, diagonal));
    // after to call
    THES_APPLY_VALUED_RETURN(Ret, outer_lambda(false_tag, true_tag, false_tag));

    THES_RETURN_EMPTY_OPTIONAL(Ret);
  } else {
    using Ret = thes::TypeSeq<decltype(outer_lambda(true_tag, true_tag, true_tag)),
                              decltype(outer_lambda(false_tag, true_tag, true_tag)),
                              decltype(diagonal_op(idx, diagonal))>::Unique;

    // before
    THES_APPLY_VALUED_RETURN(Ret, outer_lambda(true_tag, true_tag, true_tag));
    // after
    THES_APPLY_VALUED_RETURN(Ret, outer_lambda(false_tag, true_tag, true_tag));
    // diagonal
    diagonal = valuator.diagonal(diagonal, grex::scalar_tag);
    THES_APPLY_VALUED_RETURN(Ret, diagonal_op(idx, diagonal));

    THES_RETURN_EMPTY_OPTIONAL(Ret);
  }
}

template<typename TIdx, typename TValuator, typename TBeforeOp, typename TAfterOp,
         grex::AnyVectorTag TTag>
THES_ALWAYS_INLINE inline constexpr void
banded_iterate(const TIdx idx, const auto pos, const TValuator& valuator, TBeforeOp before_op,
               auto diagonal_op, TAfterOp after_op, UnorderedTag /*is_ordered*/, TTag vec_tag) {
  using Real = TValuator::Real;
  using Size = TValuator::Size;
  using Vec = grex::Vector<Real, TTag::size>;

  constexpr auto vec_size = TTag::size;
  constexpr auto dimension_num = TValuator::dimension_num;
  constexpr auto last_dim = dimension_num - 1;

  auto val_tag = vec_tag.instantiate(grex::type_tag<Real>);

  decltype(auto) sys_info = valuator.info();
  assert(std::get<last_dim>(pos) + vec_tag.part() <= sys_info.axis_size(thes::auto_tag<last_dim>));

  const auto cell_infos = valuator.begin() + index_value(idx);
  const auto cell_info = cell_infos.load_ext(vec_tag);
  // ASSUMPTION: cell_value & (~vector_info.mask) = 0
  const auto cell_value = valuator.cell_value(cell_info, val_tag);

  Vec diagonal = valuator.diagonal_base(cell_value, val_tag);

  auto update_for_connection = [&](auto j, auto side, auto op, auto off_op, Size off,
                                   auto tag) THES_ALWAYS_INLINE {
    const auto side_info = off_op(cell_infos, off).load_ext(vec_tag);
    // ASSUMPTIONS:
    // * tag.mask()[i] = 0 → diff_coeff[i] = 0
    // * diff_coeff[i] = 0 → off_diagonal[i] = 0 and diagonal_summand[i] = 0
    const auto diff_coeff = valuator.template connection_value<j, side>(cell_info, side_info, tag);
    op(
      valuator.template off_diagonal<j, side>(diff_coeff, tag),
      [&](auto it) THES_ALWAYS_INLINE { return off_op(it, off); }, j);
    diagonal += valuator.template diagonal_summand<j, side>(diff_coeff, tag);
  };

  // j = 0, …, dimensions - 2: The index holds for the entire part
  thes::star::iota<0, last_dim> |
    thes::star::for_each([&](thes::AnyIndexTag auto j) THES_ALWAYS_INLINE {
      const Size pos_j = std::get<j>(pos);
      const Size offset = sys_info.after_size(j);

      if (pos_j > 0) [[likely]] {
        update_for_connection(j, thes::auto_tag<AxisSide::START>, before_op, std::minus{}, offset,
                              val_tag);
      } else if constexpr (TValuator::non_zero_borders | thes::star::contains(j.value)) {
        // ASSUMPTION: cell_value = 0 → border_diagonal_summand = 0
        diagonal +=
          valuator.template border_diagonal_summand<j, AxisSide::START>(cell_value, val_tag);
      }

      if (pos_j + 1 < sys_info.axis_size(j)) [[likely]] {
        update_for_connection(j, thes::auto_tag<AxisSide::END>, after_op, std::plus{}, offset,
                              val_tag);
      } else if constexpr (TValuator::non_zero_borders | thes::star::contains(j.value)) {
        diagonal +=
          valuator.template border_diagonal_summand<j, AxisSide::END>(cell_value, val_tag);
      }
    });

  // j = dimensions - 1
  {
    const Size pos_back = std::get<last_dim>(pos);
    const auto axis_index = [&]() THES_ALWAYS_INLINE {
      if constexpr (TValuator::non_zero_borders | thes::star::contains(last_dim)) {
        return sys_info.axis_index_vector(pos_back, last_dim, vec_tag);
      } else {
        return thes::Empty{};
      }
    }();

    {
      if (pos_back == 0) [[unlikely]] {
        if constexpr (TValuator::non_zero_borders | thes::star::contains(last_dim)) {
          const auto start_mask = axis_index == 0;
          const auto start_vector =
            valuator.template border_diagonal_summand<last_dim, AxisSide::START>(cell_value,
                                                                                 val_tag);
          diagonal = grex::mask_add(start_mask, diagonal, start_vector, val_tag);
        }

        const auto mask = val_tag.mask().insert(grex::index_tag<0>, false);
        update_for_connection(thes::index_tag<last_dim>, thes::auto_tag<AxisSide::START>, before_op,
                              std::minus{}, 1, grex::typed_masked_tag(mask));
      } else {
        update_for_connection(thes::index_tag<last_dim>, thes::auto_tag<AxisSide::START>, before_op,
                              std::minus{}, 1, val_tag);
      }
    }
    {
      const auto dim = sys_info.axis_size(thes::auto_tag<last_dim>);
      if (pos_back + vec_size >= dim) [[unlikely]] {
        if constexpr (TValuator::non_zero_borders | thes::star::contains(last_dim)) {
          const auto end_mask = axis_index + 1 == sys_info.axis_size(thes::auto_tag<last_dim>);
          const auto end_vector =
            valuator.template border_diagonal_summand<last_dim, AxisSide::END>(cell_value, val_tag);
          diagonal = grex::mask_add(end_mask, diagonal, end_vector, val_tag);
        }

        assert(dim >= pos_back + 1);
        const auto part = dim - pos_back - 1;
        const auto mask_tag = grex::part_tag<vec_size>(part).instantiate(grex::type_tag<Real>);
        update_for_connection(thes::index_tag<last_dim>, thes::auto_tag<AxisSide::END>, after_op,
                              std::plus{}, 1, mask_tag);
      } else {
        update_for_connection(thes::index_tag<last_dim>, thes::auto_tag<AxisSide::END>, after_op,
                              std::plus{}, 1, val_tag);
      }
    }
  }

  diagonal = valuator.diagonal(diagonal, vec_tag);
  diagonal_op(diagonal, std::identity{});
}

template<typename TValuator>
THES_ALWAYS_INLINE inline constexpr TValuator::Real diagonal(const auto idx, const auto pos,
                                                             const TValuator& valuator) {
  using Real = TValuator::Real;
  Real diagonal = std::numeric_limits<Real>::signaling_NaN();
  iterate(
    idx, pos, [](auto...) THES_ALWAYS_INLINE { return true; }, valuator, thes::NoOp{},
    [&](auto /*i*/, auto v) THES_ALWAYS_INLINE { diagonal = v; }, thes::NoOp{}, valued_tag,
    unordered_tag);
  assert(std::isfinite(diagonal));
  return diagonal;
}

template<typename TIdx, typename TValuator>
THES_ALWAYS_INLINE inline constexpr auto
offdiagonal(const TIdx row_idx, const auto pos, const TValuator& valuator, const TIdx col_idx) {
  using Real = TValuator::Real;
  using Optional = std::optional<Real>;

  auto lambda = [&](TIdx index, Real value) THES_ALWAYS_INLINE {
    if (index == col_idx) [[likely]] {
      return Optional{value};
    }
    return Optional{};
  };
  return iterate(
    row_idx, pos, [](auto...) THES_ALWAYS_INLINE { return true; }, valuator, lambda,
    thes::NoOp<Optional>{}, lambda, valued_tag, unordered_tag);
}

template<typename TIdx, typename TValuator>
THES_ALWAYS_INLINE inline constexpr TValuator::Size
offdiagonal_index_within(const TIdx idx, const auto pos, const TValuator& valuator,
                         const auto column, auto is_column_counted) {
  using Size = TValuator::Size;
  using Optional = std::optional<Size>;

  Size column_index = 0;
  auto lambda = [&](TIdx index) THES_ALWAYS_INLINE {
    if (index == column) [[likely]] {
      return Optional{column_index};
    }
    column_index += is_column_counted(index);
    return Optional{};
  };
  auto res = iterate(
    idx, pos, [](auto...) THES_ALWAYS_INLINE { return true; }, valuator, lambda,
    thes::NoOp<Optional>{}, lambda, unvalued_tag, unordered_tag);
  assert(res.has_value());
  return *res;
}
} // namespace lineal::stencil

#endif // INCLUDE_LINEAL_TENSOR_DYNAMIC_STENCIL_MATRIX_ALGORITHM_HPP
