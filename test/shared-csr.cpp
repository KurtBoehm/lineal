// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <cassert>
#include <cmath>
#include <utility>
#include <vector>

#include "thesauros/charconv.hpp"
#include "thesauros/format.hpp"
#include "thesauros/ranges.hpp"
#include "thesauros/static-ranges.hpp"
#include "thesauros/test.hpp"
#include "thesauros/types.hpp"
#include "thesauros/utility.hpp"

#include "lineal/base.hpp"
#include "lineal/component-wise.hpp"
#include "lineal/environment.hpp"
#include "lineal/linear-solver.hpp"
#include "lineal/tensor.hpp"

#include "aux/default.hpp"
#include "aux/iterative.hpp"
#include "aux/materials.hpp"
#include "aux/msg.hpp"
#include "aux/test.hpp"

namespace msg = lineal::msg;
namespace test = lineal::test;

using Tlax = test::DefaultSharedDefs<>;

using Defs = Tlax::Defs;
using Real = Defs::Real;
using LoReal = Defs::LoReal;
using Size = Defs::Size;
using SizeByte = Defs::SizeByte;

using Mat = Tlax::CsrMatrix<LoReal>;
using Vec = Tlax::DenseVector<Real>;
using LoVec = Tlax::DenseVector<LoReal>;
using Value = Mat::Value;

int main() {
  const Size dim = 73;

  auto row_op = [&](Size row, auto op) {
    if (row > 0) {
      op(row - 1, -Value(row));
    }
    op(row, Value(2 * (row + 1)) + Value(0.1));
    if (row + 1 < dim) {
      op(row + 1, -Value(row + 2));
    }
  };

  const auto mat = [&] {
    Mat::RowWiseBuilder builder{};
    builder.initialize(dim, lineal::IsSymmetric{false});

    Mat::RowOffsets row_offsets_ref{0};
    Mat::ColumnIndices columns_ref{};
    Mat::Entries values_ref{};

    auto insert = [&](Size col, Value value) {
      builder.insert(col, value);
      columns_ref.push_back(col);
      values_ref.push_back(value);
    };
    auto row_inc = [&] {
      ++builder;
      row_offsets_ref.push_back(static_cast<Size>(values_ref.size()));
    };

    for (const auto i : thes::range(dim)) {
      row_op(i, insert);
      row_inc();
    }
    const auto out = std::move(builder).build();
    fmt::print("{}×{}\n", thes::numeric_string(out.row_num())->view(),
               thes::numeric_string(out.column_num())->view());

    decltype(auto) row_offsets = out.row_offsets_extended();
    THES_ALWAYS_ASSERT(thes::test::range_eq(row_offsets, row_offsets_ref));

    return out;
  }();

  for (auto row : mat) {
    std::vector<Size> row_cols{};
    std::vector<Value> row_vals{};
    row.iterate(
      [&](auto col, auto value) {
        row_cols.push_back(col);
        row_vals.push_back(value);
      },
      lineal::valued_tag, lineal::unordered_tag);

    THES_ALWAYS_ASSERT(thes::test::range_eq(row, row_vals));
    THES_ALWAYS_ASSERT(thes::test::range_eq(
      thes::transform_range([](auto it) { return it.index(); }, thes::iter_range(row)), row_cols));

    row.iterate(
      [&](auto idx, auto val) {
        THES_ALWAYS_ASSERT(Value(idx + 1) == -val && idx + 1 == row.index());
      },
      [&](auto idx, auto val) {
        THES_ALWAYS_ASSERT(2 * Value(idx + 1) + Value(0.1) == val && idx == row.index());
      },
      [&](auto idx, auto val) {
        THES_ALWAYS_ASSERT(Value(idx + 1) == -val && idx == row.index() + 1);
      },
      lineal::valued_tag, lineal::unordered_tag);
  }

  const auto env = Tlax::make_env(2U);
  decltype(auto) expo = env.execution_policy();

  LoVec factor_vec(dim, 1, env);

  auto prod = mat * factor_vec;
  auto prod_vec = lineal::create_from<LoVec>(prod, expo);
  THES_ALWAYS_ASSERT(test::vector_eq(prod_vec, prod));

  const auto ref_prod = thes::transform_range(
    [&](auto idx) {
      Real sum = 0;
      row_op(idx, [&](Size j, LoReal v) { sum += Real(v) * Real(factor_vec[j]); });
      return LoReal(sum);
    },
    thes::range(dim));
  THES_ALWAYS_ASSERT(test::vector_eq(prod_vec, ref_prod));

  {
    const auto envi = env.add_object("lu");

    lineal::LuSolver<Real, Mat> lu_solver{};
    auto lu_inst = lu_solver.instantiate(mat, envi);

    Vec sol(mat.row_num());
    Vec aux(mat.row_num());
    lu_inst.solve(sol, prod_vec, thes::Tuple{aux}, envi);

    const auto err = lineal::euclidean_norm<Real>(sol - factor_vec, expo);
    envi.log("error", err);
    THES_ALWAYS_ASSERT(err <= 7e-6);
  }

  auto solver_tests = [&](thes::TypedValueTag<lineal::IsSymmetric> auto symmetric) {
    const auto envi =
      env.add_object((symmetric == lineal::IsSymmetric{true}) ? "solver_sym" : "solver_nonsym");

    const auto wright_instance = Tlax::make_wright<Vec, Mat, Vec, symmetric>();
    envi.log("coarse_solver", thes::type_name(wright_instance.coarse_solver()));
    THES_ALWAYS_ASSERT((test::direct<Real, Vec>(mat, wright_instance, envi) <= 1e-13));

    {
      Vec sol(mat.row_num());

      std::vector<Real> residuals{
        // regular
        /*bicgstab=*/0.1,
        /*sor=*/1,
        /*ssor1=*/1,
        /*sor5=*/0.75,
        /*ssor5=*/0.3,
        /*pbicgstab_ssor1=*/1e-11,
        /*pbicgstab_amg=*/1e-12,
        // ultra
        /*bicgstab=*/0.1,
        /*sor=*/1,
        /*ssor1=*/1,
        /*sor5=*/0.75,
        /*ssor5=*/0.3,
        /*pbicgstab_ssor1=*/1e-11,
        /*pbicgstab_amg=*/1e-12,
      };
      auto it = residuals.begin();

      auto op = [&](auto res, auto& /*sol*/, const auto& envii) {
        const auto bound = *(it++);
        const auto err_max = lineal::max_norm<Real>(sol - factor_vec, expo);
        envii.log("error", msg::ResidualErrorBound{res, err_max, bound});
        THES_ALWAYS_ASSERT(res <= bound && err_max <= bound);
      };
      test::suite<Defs>(mat, sol, prod_vec, wright_instance, envi, op);
    }

    {
      using Valuator = Tlax::LookupValuatorZero;
      using SysInfo = Tlax::LookupSymmetricDiffInfo;
      constexpr SysInfo::SizeArray dims{4, 7, 6};
      constexpr SysInfo::MaterialCoeffs coeffs{0.0, 1.0, 2.0, 3.0};
      constexpr SysInfo::NoneMaterialMap none_mats{false, false, false, false};

      SysInfo info{
        dims,
        dims | thes::star::transform([](Size idx) { return Real(idx); }) | thes::star::to_array,
        coeffs,
        none_mats,
        /*solution_start=*/1,
        /*solution_end=*/0,
      };

      auto material =
        lineal::materials::diagonal_lines<Defs>(info, thes::range(info.total_size()), expo);
      const Valuator valuator{info, material};
      const auto sys = lineal::csr_from_adjacent_stencil<Mat, Vec>(valuator, envi);

      {
        Vec sol(sys.rhs.size());

        std::vector<Real> residuals{
          // regular
          /*cg=*/1e-13,
          /*bicgstab=*/2e-13,
          /*sor=*/5e-4,
          /*ssor1=*/1e-6,
          /*sor5=*/1e-14,
          /*ssor5=*/1e-14,
          /*pcg_ssor1=*/1e-13,
          /*pbicgstab_ssor1=*/1e-13,
          /*pcg_amg=*/1e-13,
          /*pbicgstab_amg=*/1e-13,
          // ultra
          /*cg=*/1e-13,
          /*bicgstab=*/2e-13,
          /*sor=*/5e-4,
          /*ssor1=*/1e-6,
          /*sor5=*/1e-14,
          /*ssor5=*/1e-14,
          /*pcg_ssor1=*/1e-13,
          /*pbicgstab_ssor1=*/1e-13,
          /*pcg_amg=*/1e-13,
          /*pbicgstab_amg=*/1e-13,
        };
        auto it = residuals.begin();

        auto op = [&, effdiff = lineal::EffDiffCalc{valuator, envi}](auto res, auto& /*sol*/,
                                                                     const auto& envii) {
          const auto bound = *(it++);
          const auto err_max = lineal::max_norm<Real>(sys.lhs * sol - sys.rhs, expo);
          envii.log("error", msg::ResidualsBound{res, err_max, bound});
          THES_ALWAYS_ASSERT(res <= bound && err_max <= bound);

          const auto eff_diff = effdiff(valuator, sys.compress_map, sol, envii);
          envii.log("eff_diff", msg::RefValue{eff_diff, 1.5});
          THES_ALWAYS_ASSERT(std::abs(eff_diff - 1.5) <= std::pow(bound, 1.0 / 3.0));
        };
        test::suite<Defs>(sys.lhs, sol, sys.rhs, wright_instance, envi, op);
      }
    }
  };
  solver_tests(thes::auto_tag<lineal::IsSymmetric{true}>);
  solver_tests(thes::auto_tag<lineal::IsSymmetric{false}>);
}
