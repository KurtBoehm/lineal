// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_LINEAR_SOLVER_CHOLESKY_DECOMPOSE_HPP
#define INCLUDE_LINEAL_LINEAR_SOLVER_CHOLESKY_DECOMPOSE_HPP

#include <cassert>

#include "thesauros/functional.hpp"
#include "thesauros/macropolis.hpp"
#include "thesauros/math.hpp"
#include "thesauros/ranges.hpp"
#include "thesauros/types.hpp"

#include "lineal/base.hpp"

namespace lineal {
template<typename TReal, SharedMatrix TMat, SharedMatrix TCholeskyMat>
struct CholeskyDecomposer {
  using Real = TReal;
  using Size = thes::TypeSeq<typename TMat::Size, typename TCholeskyMat::Size>::Unique;
  using RowIdx = thes::TypeSeq<typename TMat::RowIdx, typename TCholeskyMat::RowIdx>::Unique;
  using ColumnIdx =
    thes::TypeSeq<typename TMat::ColumnIdx, typename TCholeskyMat::ColumnIdx>::Unique;

  using OutValue = TCholeskyMat::Value;

  static constexpr Real dot(const auto& row1, const auto& row2) {
    Real dot_prod = 0;

    auto it1 = row1.begin();
    const auto end1 = row1.end();
    for (const auto [idx2, val2] : row2) {
      for (; it1 != end1 && it1.index() < idx2; ++it1) {
      }
      if (it1 == end1) {
        break;
      }
      if (it1.index() == idx2) {
        dot_prod += Real(*it1) * Real(val2);
      }
    }

    return dot_prod;
  }

  // An implementation of the Cholesky–Banachiewicz algorithm.
  static constexpr TCholeskyMat decompose(const TMat& matrix) {
    using Builder = TCholeskyMat::RowWiseBuilder;
    assert(matrix.row_num() == matrix.column_num());

    Builder lower_builder{thes::void_storage_cref(matrix.distributed_info_storage())};

    lower_builder.initialize(matrix.row_num(), IsSymmetric{false});
    for (const Size i : thes::range(matrix.row_num())) {
      // matrix[i][j] = base - dot(matrix[j], matrix[i])
      auto lambda = [&lower_builder](ColumnIdx j, Real base) THES_ALWAYS_INLINE {
        auto row_j = lower_builder[RowIdx{index_value(j)}];
        assert(row_j.back_column() == j);
        const Real value = (base - dot(row_j, lower_builder.row_columns())) / Real(row_j.back());
        if (value != Real{0}) {
          lower_builder.insert(j, static_cast<OutValue>(value));
        }
      };

      Size next_start = 0;
      bool is_first = true;
      Real diag = 0;
      matrix[RowIdx{i}].iterate(
        [&](ColumnIdx j, Real value) {
          const auto jval = index_value(j);
          if (is_first) {
            is_first = false;
          } else {
            for (const auto k : thes::range(next_start, jval)) {
              lambda(ColumnIdx{k}, 0);
            }
          }
          lambda(j, value);
          next_start = jval + 1;
        },
        [&diag](ColumnIdx /*j*/, Real value) THES_ALWAYS_INLINE { diag = value; }, thes::NoOp{},
        valued_tag, ordered_tag);
      for (const Size j : thes::range(next_start, i)) {
        lambda(ColumnIdx{j}, 0);
      }

      for (const auto [key, value] : lower_builder.row_columns()) {
        diag -= Real(value) * Real(value);
      }
      assert(diag >= 0);
      lower_builder.insert(ColumnIdx{i}, static_cast<OutValue>(thes::fast::sqrt(diag)));
      ++lower_builder;
    }
    return std::move(lower_builder).build();
  }
};

template<typename TReal, SharedMatrix TCholeskyMat, SharedMatrix TMat>
inline constexpr TCholeskyMat cholesky_decompose(const TMat& mat) {
  return CholeskyDecomposer<TReal, TMat, TCholeskyMat>::decompose(mat);
}
} // namespace lineal

#endif // INCLUDE_LINEAL_LINEAR_SOLVER_CHOLESKY_DECOMPOSE_HPP
