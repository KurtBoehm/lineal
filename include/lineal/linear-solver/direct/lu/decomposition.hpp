// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_LINEAR_SOLVER_DIRECT_LU_DECOMPOSITION_HPP
#define INCLUDE_LINEAL_LINEAR_SOLVER_DIRECT_LU_DECOMPOSITION_HPP

#include <algorithm>
#include <cassert>
#include <utility>

#include "ankerl/unordered_dense.h"
#include "thesauros/types.hpp"

#include "lineal/base.hpp"

namespace lineal {
template<typename TReal, SharedMatrix TMat, SharedMatrix TLuMat>
struct LuDecomposer {
  using Real = TReal;
  using Size = thes::TypeSeq<typename TMat::Size, typename TLuMat::Size>::Unique;
  using RowIdx = thes::TypeSeq<typename TMat::RowIdx, typename TLuMat::RowIdx>::Unique;
  using ColumnIdx = thes::TypeSeq<typename TMat::ColumnIdx, typename TLuMat::ColumnIdx>::Unique;

  using OutValue = TLuMat::Value;

  static constexpr std::pair<TLuMat, TLuMat> decompose(const TMat& matrix) {
    using Builder = TLuMat::RowWiseBuilder;
    assert(matrix.row_num() == matrix.column_num());

    Builder lbld{thes::void_storage_cref(matrix.distributed_info_storage())};
    lbld.initialize(matrix.row_num(), IsSymmetric{false});
    Builder ubld{thes::void_storage_cref(matrix.distributed_info_storage())};
    ubld.initialize(matrix.row_num(), IsSymmetric{false});

    // TODO merge with the column indices?
    // that would add at least one case distinction in return for reducing memory consumption
    ankerl::unordered_dense::map<Size, Real> new_values{};
    ankerl::unordered_dense::set<Size> l_new_cidxs{};
    ankerl::unordered_dense::set<Size> u_new_cidxs{};

    for (auto row : matrix) {
      const auto i = index_value(row.index());

      row.iterate(
        [&](ColumnIdx j, auto v) {
          const auto jval = index_value(j);
          new_values.emplace(jval, Real(v));
          if (jval < i) {
            l_new_cidxs.emplace(jval);
          } else {
            u_new_cidxs.emplace(jval);
          }
        },
        valued_tag, ordered_tag);

      while (!l_new_cidxs.empty()) {
        // get the smallest column index in l_new_cidxs
        const auto lj_it = std::ranges::min_element(l_new_cidxs);
        const Size lj = l_new_cidxs.extract(lj_it);

        const auto uj_row = ubld[RowIdx{lj}];
        const auto uj_it = uj_row.begin();
        assert(uj_it.index() == ColumnIdx{lj});
        const auto ujdiag = *uj_it;
        const Real lval = new_values.extract(lj).value().second / Real(ujdiag);
        lbld.insert(ColumnIdx{lj}, OutValue(lval));

        // TODO split into ante-diagonal/diagonal/post-diagonal functions?
        // this increases the generate code size, but removes the inner if
        bool first = true;
        uj_row.iterate(
          [&](ColumnIdx uj, auto uv) {
            const auto ujval = index_value(uj);
            if (first) {
              first = false;
              return;
            }
            const Real sub = lval * Real(uv);
            auto it = new_values.find(ujval);
            if (it != new_values.end()) {
              it->second -= sub;
            } else {
              new_values.emplace(ujval, -sub);

              if (ujval < i) {
                l_new_cidxs.emplace(ujval);
              } else {
                u_new_cidxs.emplace(ujval);
              }
            }
          },
          lineal::valued_tag, lineal::unordered_tag);
      }

      // TODO avoid storing this and implicitly assume it is there?
      lbld.insert(ColumnIdx{i}, 1);
      ++lbld;

      for (const Size cidx : u_new_cidxs) {
        ubld.insert(ColumnIdx{cidx}, OutValue(new_values[cidx]));
      }
      ++ubld;

      new_values.clear();
      u_new_cidxs.clear();
    }
    return {std::move(lbld).build(), std::move(ubld).build()};
  }
};

template<typename TReal, SharedMatrix TLuMat, SharedMatrix TMat>
constexpr std::pair<TLuMat, TLuMat> lu_decompose(const TMat& mat) {
  return LuDecomposer<TReal, TMat, TLuMat>::decompose(mat);
}

} // namespace lineal

#endif // INCLUDE_LINEAL_LINEAR_SOLVER_DIRECT_LU_DECOMPOSITION_HPP
