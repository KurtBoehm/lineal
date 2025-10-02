// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <cassert>
#include <cstddef>
#include <utility>
#include <vector>

#include "thesauros/charconv.hpp"
#include "thesauros/format.hpp"
#include "thesauros/ranges.hpp"
#include "thesauros/test.hpp"
#include "thesauros/types.hpp"
#include "thesauros/utility.hpp"

#include "lineal/base.hpp"
#include "lineal/component-wise.hpp"
#include "lineal/environment.hpp"
#include "lineal/io.hpp"
#include "lineal/linear-solver.hpp"
#include "lineal/tensor.hpp"

#include "aux/default.hpp"
#include "aux/iterative.hpp"
#include "aux/msg.hpp"
#include "aux/test.hpp"

namespace fix = lineal::fix;
namespace msg = lineal::msg;
namespace test = lineal::test;
namespace compat = lineal::compat;

inline constexpr std::size_t block_size = 3;
using Tlax = test::DefaultSharedDefs<block_size>;

using Defs = Tlax::Defs;
using Real = Defs::Real;
using LoReal = Defs::LoReal;
using Size = Defs::Size;
using SizeByte = Defs::SizeByte;

using HiBMat = fix::DenseMatrix<Real, block_size, block_size>;
using LoBMat = fix::DenseMatrix<LoReal, block_size, block_size>;
using HiBVec = fix::DenseVector<Real, block_size>;
using LoBVec = fix::DenseVector<LoReal, block_size>;

using HiMat = Tlax::CsrMatrix<HiBMat>;
using LoMat = Tlax::CsrMatrix<LoBMat>;
using HiVec = Tlax::DenseVector<HiBVec>;
using LoVec = Tlax::DenseVector<LoBVec>;

int main() {
  const Size dim = 73;

  auto bdia = [&](LoReal v) { return LoBMat::diagonal(LoBVec(v)); };
  auto row_op = [&](Size row, auto op) {
    using Scalar = LoReal;
    if (row > 0) {
      op(row - 1, bdia(-Scalar(row)));
    }
    op(row, bdia(Scalar(2 * (row + 1)) + Scalar(0.1)));
    if (row + 1 < dim) {
      op(row + 1, bdia(-Scalar(row + 2)));
    }
  };

  const auto mat = [&] {
    using Value = LoMat::Value;

    LoMat::RowWiseBuilder builder{};
    builder.initialize(dim, lineal::IsSymmetric{false});

    LoMat::RowOffsets row_offsets_ref{0};
    LoMat::ColumnIndices columns_ref{};
    LoMat::Entries values_ref{};

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
    using Value = LoMat::Value;
    using Scalar = lineal::ScalarType<Value>;

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
        THES_ALWAYS_ASSERT(bdia(Scalar(idx + 1)) == -val && idx + 1 == row.index());
      },
      [&](auto idx, auto val) {
        THES_ALWAYS_ASSERT(bdia(2 * Scalar(idx + 1) + Scalar(0.1)) == val && idx == row.index());
      },
      [&](auto idx, auto val) {
        THES_ALWAYS_ASSERT(bdia(Scalar(idx + 1)) == -val && idx == row.index() + 1);
      },
      lineal::valued_tag, lineal::unordered_tag);
  }

  const auto env = Tlax::make_env(1U);
  decltype(auto) expo = env.execution_policy();

  LoVec factor_vec(dim, LoBVec(1), env);

  auto prod = mat * factor_vec;
  auto prod_vec = lineal::create_from<LoVec>(prod, expo);
  THES_ALWAYS_ASSERT(test::vector_eq(prod_vec, prod));

#if false
  fmt::print("\n");
  fmt::print("lhs:\n{}\n", lineal::matrix_print(mat, lineal::unskip_zero_tag, thes::format_tag,
                                                lineal::ordered_tag));
  fmt::print("rhs:\n{}\n", lineal::vector_print(prod_vec));
#endif

  {
    const auto ref_prod = thes::transform_range(
      [&](auto idx) {
        HiBVec sum = HiBVec::zero();
        row_op(idx, [&](Size j, LoBMat v) {
          sum += compat::cast<Real>(v) * compat::cast<Real>(factor_vec[j]);
        });
        return compat::cast<LoReal>(sum);
      },
      thes::range(dim));
    THES_ALWAYS_ASSERT(test::vector_eq(prod_vec, ref_prod));
  }

  {
    const auto envi = env.add_object("lu");

    lineal::LuSolver<Real, HiMat> lu_solver{};
    auto lu_inst = lu_solver.instantiate(mat, envi);

    HiVec sol(mat.row_num());
    HiVec aux(mat.row_num());
    lu_inst.solve(sol, prod_vec, thes::Tuple{aux}, envi);

    const auto err = lineal::euclidean_norm<Real>(sol - factor_vec, expo);
    envi.log("error", err);
    THES_ALWAYS_ASSERT(err <= 1e-13);
  }

  {
    using BMat = fix::DenseMatrix<Real, 2, 2>;
    using Mat = Tlax::CsrMatrix<BMat>;
    using BVec = fix::DenseVector<Real, 2>;
    using Vec = Tlax::DenseVector<BVec>;
    const auto envi = env.add_object("cholesky");
    auto mat4 = [&](BMat a00, BMat a01, BMat a10, BMat a11) {
      Mat::RowWiseBuilder builder{};
      builder.initialize(2, lineal::IsSymmetric{false});
      builder.insert(0, a00);
      builder.insert(1, a01);
      ++builder;
      builder.insert(0, a10);
      builder.insert(1, a11);
      ++builder;
      return std::move(builder).build();
    };
    auto vec2 = [&](BVec v0, BVec v1) {
      Vec ref_sol(2);
      ref_sol[Size{0}] = v0;
      ref_sol[Size{1}] = v1;
      return ref_sol;
    };

    const auto cmat =
      mat4(BMat{{4, 1}, {1, 5}}, BMat{{2, 3}, {1, 2}}, BMat{{2, 1}, {3, 2}}, BMat{{6, 1}, {1, 7}});
    fmt::print("\nMatrix:\n{}", lineal::matrix_print(cmat, lineal::unskip_zero_tag,
                                                     thes::format_tag, lineal::ordered_tag));
    lineal::CholeskySolver<Real, Mat> solver{};
    auto sinst = solver.instantiate(cmat, envi);
    const auto& lower = sinst.lower();
    fmt::print("\nCholesky:\n{}", lineal::matrix_print(lower, lineal::unskip_zero_tag,
                                                       thes::format_tag, lineal::ordered_tag));
    fmt::print("\nC·C^T:\n{}, {}\n{} {}", lower[0][0] * fix::transpose(lower[0][0]),
               lower[0][0] * fix::transpose(lower[1][0]), lower[1][0] * fix::transpose(lower[0][0]),
               lower[1][0] * fix::transpose(lower[1][0]) +
                 lower[1][1] * fix::transpose(lower[1][1]));

    Vec sol(2);
    Vec rhs = vec2(BVec{11, 14}, BVec{11, 15});
    Vec aux(2);
    sinst.solve(sol, rhs, thes::Tuple{aux}, envi);
    fmt::print("\nsolution: {}", sol);
    const Vec ref = vec2(BVec{1, 2}, BVec{1, 1});
    const auto defect = lineal::euclidean_norm<Real>(sol - ref, envi.execution_policy());
    fmt::print("\ndefect: {}", defect);
    THES_ALWAYS_ASSERT(defect < 1e-15);
  }

  {
    using BMat = fix::DenseMatrix<Real, 2, 2>;
    using Mat = Tlax::CsrMatrix<BMat>;
    using BVec = fix::DenseVector<Real, 2>;
    using Vec = Tlax::DenseVector<BVec>;
    const auto envi = env.add_object("cholesky");
    auto mat9 = [&](BMat a00, BMat a01, BMat a02, BMat a10, BMat a11, BMat a12, BMat a20, BMat a21,
                    BMat a22) {
      Mat::RowWiseBuilder builder{};
      builder.initialize(3, lineal::IsSymmetric{false});
      builder.insert(0, a00);
      builder.insert(1, a01);
      builder.insert(2, a02);
      ++builder;
      builder.insert(0, a10);
      builder.insert(1, a11);
      builder.insert(2, a12);
      ++builder;
      builder.insert(0, a20);
      builder.insert(1, a21);
      builder.insert(2, a22);
      ++builder;
      return std::move(builder).build();
    };
    auto vec3 = [&](BVec v0, BVec v1, BVec v2) {
      Vec ref_sol(3);
      ref_sol[Size{0}] = v0;
      ref_sol[Size{1}] = v1;
      ref_sol[Size{2}] = v2;
      return ref_sol;
    };

    const auto cmat =
      mat9(BMat{{6, 1}, {1, 7}}, BMat{{2, 3}, {1, 2}}, BMat{{4, 5}, {3, 4}}, // row 0
           BMat{{2, 1}, {3, 2}}, BMat{{8, 1}, {1, 9}}, BMat{{2, 3}, {1, 2}}, // row 1
           BMat{{4, 3}, {5, 4}}, BMat{{2, 1}, {3, 2}}, BMat{{10, 1}, {1, 11}} // row 2
      );
    fmt::print("\nMatrix:\n{}", lineal::matrix_print(cmat, lineal::unskip_zero_tag,
                                                     thes::format_tag, lineal::ordered_tag));

    lineal::CholeskySolver<Real, Mat> chsolver{};
    auto chinst = chsolver.instantiate(cmat, envi);
    const auto& chlower = chinst.lower();
    fmt::print("\nCholesky:\n{}", lineal::matrix_print(chlower, lineal::unskip_zero_tag,
                                                       thes::format_tag, lineal::ordered_tag));

    lineal::LuSolver<Real, Mat> lusolver{};
    auto luinst = lusolver.instantiate(cmat, envi);
    const auto& lulower = luinst.lower();
    const auto& luupper = luinst.upper();
    fmt::print(
      "\nLU:\n{}\n{}",
      lineal::matrix_print(lulower, lineal::unskip_zero_tag, thes::format_tag, lineal::ordered_tag),
      lineal::matrix_print(luupper, lineal::unskip_zero_tag, thes::format_tag,
                           lineal::ordered_tag));

    const Vec rhs = vec3(BVec{32, 30}, BVec{34, 35}, BVec{38, 30});
    const Vec ref = vec3(BVec{3, 3}, BVec{3, 2}, BVec{1, -1});
    {
      Vec sol(3);
      lineal::assign(sol, rhs - 0.5 * rhs, envi.execution_policy());
      THES_ALWAYS_ASSERT(test::vector_eq(sol, 0.5 * rhs));
    }
    {
      Vec sol(3);
      Vec aux(3);
      chinst.solve(sol, rhs, thes::Tuple{aux}, envi);
      fmt::print("\nCholesky solution: {}", sol);
      const auto defect = lineal::euclidean_norm<Real>(sol - ref, envi.execution_policy());
      fmt::print("\ndefect: {}", defect);
      THES_ALWAYS_ASSERT(defect < 1e-14);
    }
    {
      Vec sol(3);
      Vec aux(3);
      luinst.solve(sol, rhs, thes::Tuple{aux}, envi);
      fmt::print("\nLU solution: {}", sol);
      const auto defect = lineal::euclidean_norm<Real>(sol - ref, envi.execution_policy());
      fmt::print("\ndefect: {}", defect);
      THES_ALWAYS_ASSERT(defect < 1e-14);
    }
    {
      using SorSolver = lineal::SorSolver<Real, void, lineal::forward_tag, lineal::regular_sor>;
      SorSolver sorsolver{/*relax=*/1.0};
      auto sorinst = sorsolver.instantiate(cmat, envi);

      Vec sol(3, BVec::zero(), env);
      Vec aux(3);
      for (std::size_t i = 0; i < 256; ++i) {
        sorinst.apply(sol, rhs, thes::Tuple{aux}, envi);
      }
      fmt::print("\nSOR solution: {}", sol);
      const auto defect = lineal::euclidean_norm<Real>(sol - ref, envi.execution_policy());
      fmt::print("\ndefect: {}", defect);
      THES_ALWAYS_ASSERT(defect < 2e-9);
    }
    {
      using SorSolver = lineal::SsorSolver<Real, void, lineal::regular_sor>;
      SorSolver sorsolver{/*relax=*/1.0};
      auto sorinst = sorsolver.instantiate(cmat, envi);

      Vec sol(3, BVec::zero(), env);
      Vec aux(3);
      for (std::size_t i = 0; i < 512; ++i) {
        sorinst.apply(sol, rhs, thes::Tuple{aux}, envi);
      }
      fmt::print("\nSSOR solution: {}", sol);
      const auto defect = lineal::euclidean_norm<Real>(sol - ref, envi.execution_policy());
      fmt::print("\ndefect: {}", defect);
      THES_ALWAYS_ASSERT(defect < 1e-7);
    }
  }

  auto solver_tests = [&](thes::TypedValueTag<lineal::IsSymmetric> auto symmetric) {
    const auto envi =
      env.add_object((symmetric == lineal::IsSymmetric{true}) ? "solver_sym" : "solver_nonsym");

    const auto wright_instance =
      Tlax::make_wright<HiVec, LoMat, LoVec, symmetric, lineal::SorVariant::regular,
                        lineal::amg::FrobeniusTransform<Real>>();
    envi.log("coarse_solver", thes::type_name(wright_instance.coarse_solver()));
    THES_ALWAYS_ASSERT((test::direct<Real, HiVec>(mat, wright_instance, envi) <= 1e-13));

    {
      HiVec sol(mat.row_num());

      std::vector<Real> residuals{
        /*bicgstab=*/0.1,
        /*sor=*/1.6,
        /*ssor1=*/1,
        /*sor5=*/0.2,
        /*ssor5=*/0.01,
        /*pbicgstab_ssor1=*/1e-12,
        /*pbicgstab_amg=*/1e-12,
      };
      auto it = residuals.begin();

      auto op = [&](auto res, auto& /*sol*/, const auto& envii) {
        const auto bound = *(it++);
        const auto err_max = lineal::max_norm<Real>(sol - factor_vec, expo);
        envii.log("error", msg::ResidualErrorBound{res, err_max, bound});
        THES_ALWAYS_ASSERT(res <= bound && err_max <= bound);
      };
      test::suite<Defs, lineal::SorVariant::regular>(mat, sol, prod_vec, wright_instance, envi, op);
    }
  };
  solver_tests(thes::auto_tag<lineal::IsSymmetric{true}>);
  solver_tests(thes::auto_tag<lineal::IsSymmetric{false}>);
}
