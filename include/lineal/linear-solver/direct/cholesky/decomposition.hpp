// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_LINEAR_SOLVER_DIRECT_CHOLESKY_DECOMPOSITION_HPP
#define INCLUDE_LINEAL_LINEAR_SOLVER_DIRECT_CHOLESKY_DECOMPOSITION_HPP

#include <cassert>

#include "thesauros/functional.hpp"
#include "thesauros/macropolis.hpp"
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
  using OutScalar = ScalarType<OutValue>;
  using MatWork = WithScalarType<typename TMat::Value, Real>;

  static constexpr MatWork dot(const auto& row1, const auto& row2) {
    MatWork dot_prod = compat::zero<MatWork>();

    auto it1 = row1.begin();
    const auto end1 = row1.end();
    for (const auto [idx2, val2] : row2) {
      for (; it1 != end1 && it1.index() < idx2; ++it1) {
      }
      if (it1 == end1) {
        break;
      }
      if (it1.index() == idx2) {
        dot_prod += compat::cast<Real>(val2) * compat::transpose(compat::cast<Real>(*it1));
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
      MatWork diag = compat::zero<MatWork>();

      // matrix[i][j] = base - dot(matrix[j], matrix[i])
      auto lambda = [&](ColumnIdx j, MatWork base) THES_ALWAYS_INLINE {
        auto row_j = lower_builder[RowIdx{index_value(j)}];
        assert(row_j.back_column() == j);
        const MatWork value{compat::solve_right<Real>(
          compat::transpose(row_j.back()), base - dot(row_j, lower_builder.row_columns()))};
        if (value != compat::zero<MatWork>()) {
          diag -= value * compat::transpose(value);
          lower_builder.insert(j, compat::cast<OutScalar>(value));
        }
      };

      Size next_start = 0;
      bool is_first = true;
      matrix[RowIdx{i}].iterate(
        [&](ColumnIdx j, auto value) {
          const auto jval = index_value(j);
          if (is_first) {
            is_first = false;
          } else {
            for (const auto k : thes::range(next_start, jval)) {
              lambda(ColumnIdx{k}, compat::zero<MatWork>());
            }
          }
          lambda(j, compat::cast<Real>(value));
          next_start = jval + 1;
        },
        [&diag](ColumnIdx /*j*/, auto value)
          THES_ALWAYS_INLINE { diag += compat::cast<Real>(value); },
        thes::NoOp{}, valued_tag, ordered_tag);
      for (const Size j : thes::range(next_start, i)) {
        lambda(ColumnIdx{j}, compat::zero<MatWork>());
      }

      if constexpr (IsScalar<OutValue>) {
        assert(diag >= 0);
      } else {
        assert(diag != compat::zero<MatWork>());
      }

      lower_builder.insert(ColumnIdx{i},
                           compat::cast<OutScalar>(compat::cholesky_lower<MatWork>(diag)));
      ++lower_builder;
    }
    return std::move(lower_builder).build();
  }
};

template<typename TReal, SharedMatrix TCholeskyMat, SharedMatrix TMat>
constexpr TCholeskyMat cholesky_decompose(const TMat& mat) {
  return CholeskyDecomposer<TReal, TMat, TCholeskyMat>::decompose(mat);
}
} // namespace lineal

#endif // INCLUDE_LINEAL_LINEAR_SOLVER_DIRECT_CHOLESKY_DECOMPOSITION_HPP
