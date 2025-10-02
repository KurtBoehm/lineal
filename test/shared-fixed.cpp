// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>

#include "thesauros/format.hpp"
#include "thesauros/static-ranges.hpp"
#include "thesauros/test.hpp"
#include "thesauros/types.hpp"

#include "lineal/lineal.hpp"

namespace fix = lineal::fix;
namespace compat = lineal::compat;

template<std::size_t tDim>
using Mat = fix::DenseMatrix<double, tDim, tDim>;
template<std::size_t tDim>
using Vec = fix::DenseVector<double, tDim>;

static constexpr Mat<3> mat3x3{
  {1, 0, 0},
  {0, 2, 2},
  {0, 0, 3},
};
static constexpr Vec<3> vec3{1, 2, 3};
static constexpr auto prod3 = mat3x3 * vec3;
static constexpr Vec<3> ref3{1, 10, 9};
static_assert(Vec<3>{prod3} == ref3);
static constexpr auto sol3 = fix::solve<Vec<3>>(mat3x3, ref3);
static_assert(sol3 == vec3);

static constexpr Mat<2> mat2x2{{1, 0}, {2, 2}};
static constexpr Vec<2> vec2{1, 3};
static constexpr auto prod2 = mat2x2 * vec2;
static constexpr Vec<2> ref2{1, 8};
static_assert(Vec<2>{prod2} == ref2);
static constexpr auto sol2 = fix::solve<Vec<2>>(mat2x2, ref2);
static_assert(sol2 == vec2);

static constexpr Mat<1> mat1x1{{4}};
static constexpr Vec<1> vec1{3};
static constexpr auto prod1 = mat1x1 * vec1;
static constexpr Vec<1> ref1{12};
static_assert(Vec<1>{prod1} == ref1);
static constexpr auto sol1 = fix::solve<Vec<1>>(mat1x1, ref1);
static_assert(sol1 == vec1);

template<typename TReal, std::size_t tSize>
struct FixMatrixTester {
  using Real = TReal;
  using Mat = fix::DenseMatrix<TReal, tSize, tSize>;
  using Vec = fix::DenseVector<TReal, tSize>;

  Mat mat = thes::init_required;
  Mat inv = thes::init_required;
  Vec rhs = thes::init_required;
  Vec sol = thes::init_required;
};

template<auto tTester>
void test_base() {
  using Tester = decltype(tTester);
  using Real = Tester::Real;
  using Mat = Tester::Mat;
  using Vec = Tester::Vec;

  fmt::print("mat: {}\n", tTester.mat);

  constexpr auto inv = fix::invert<Mat>(tTester.mat);
  fmt::print("inverse: {}\n", inv);
  static_assert(fix::frobenius_norm<Real>(inv - tTester.inv) < 3e-15);

  constexpr Mat refl{inv * tTester.mat};
  fmt::print("reference left: {}\n", refl);
  static_assert(fix::frobenius_norm<Real>(refl - Mat::identity()) < 2e-14);

  constexpr Mat refr{tTester.mat * inv};
  fmt::print("reference right: {}\n", refr);
  static_assert(fix::frobenius_norm<Real>(refr - Mat::identity()) < 3e-15);

  constexpr auto sol = fix::solve<Vec>(tTester.mat, tTester.rhs);
  constexpr Vec refs{tTester.mat * sol};
  fmt::print("solution: {}\n", sol);
  static_assert(fix::euclidean_norm<Real>(sol - tTester.sol) < 1e-15);
  fmt::print("reference: {}\n", refs);
  static_assert(fix::euclidean_norm<Real>(refs - tTester.rhs) < 2e-15);

  constexpr auto msol = fix::solve<Mat>(tTester.mat, tTester.mat * fix::transpose(tTester.mat));
  fmt::print("matrix solution: {}\n", msol);
  static_assert(fix::frobenius_norm<Real>(msol - fix::transpose(tTester.mat)) < 2e-14);

  constexpr auto mrsol =
    fix::solve_right<Mat>(tTester.mat, fix::transpose(tTester.mat) * tTester.mat);
  fmt::print("matrix right solution: {}\n", mrsol);
  static_assert(fix::frobenius_norm<Real>(mrsol - fix::transpose(tTester.mat)) < 2e-14);
}

template<auto tTester>
void test_lu() {
  using Tester = decltype(tTester);
  using Real = Tester::Real;
  using Mat = Tester::Mat;

  constexpr auto lu = fix::decompose_lu<Mat, Mat>(tTester.mat);
  static_assert(fix::frobenius_norm<Real>(tTester.mat - lu.first * lu.second) < 1e-15);
  fmt::print("{}: {}, {}: {}\n", fmt::styled("L", thes::fg_green), lu.first,
             fmt::styled("U", thes::fg_green), lu.second);
  fmt::print("L·U: {}\n", lu.first * lu.second);

  constexpr auto tri_solve = [=](const auto& rhs) {
    return compat::solve_tri<Real>(lu.second,
                                   compat::solve_tri<Real>(lu.first, rhs, lineal::tri_lower_tag,
                                                           lineal::lhs_has_unit_diagonal_tag<true>),
                                   lineal::tri_upper_tag, lineal::lhs_has_unit_diagonal_tag<false>);
  };

  constexpr auto sol = tri_solve(tTester.rhs);
  fmt::print("LU solution: {}\n", sol);
  static_assert(fix::euclidean_norm<Real>(sol - tTester.sol) < 8e-15);

  constexpr auto msol = tri_solve(tTester.mat * fix::transpose(tTester.mat));
  fmt::print("LU matrix solution: {}\n", msol);
  static_assert(fix::frobenius_norm<Real>(msol - fix::transpose(tTester.mat)) < 2e-14);

  constexpr auto mrsol = compat::solve_right_tri<Real>(
    lu.first,
    compat::solve_right_tri<Real>(lu.second, fix::transpose(tTester.mat) * tTester.mat,
                                  lineal::tri_upper_tag, lineal::lhs_has_unit_diagonal_tag<false>),
    lineal::tri_lower_tag, lineal::lhs_has_unit_diagonal_tag<true>);
  fmt::print("LU right matrix solution: {}\n", mrsol);
  static_assert(fix::frobenius_norm<Real>(mrsol - fix::transpose(tTester.mat)) < 2e-14);
}

template<auto tTester>
void test_cholesky() {
  using Tester = decltype(tTester);
  using Real = Tester::Real;
  using Mat = Tester::Mat;

  constexpr auto c = fix::decompose_cholesky<Mat>(tTester.mat);
  constexpr Mat recons{c * fix::transpose(c)};
  fmt::print("lower Cholesky: {}\n", c);
  fmt::print("C·C^T: {}\n", recons);
  constexpr auto diff = fix::frobenius_norm<Real>(tTester.mat - recons);
  static_assert(diff < 2e-15);
}

int main() {
  {
    Mat<3> lhs{
      {1, 0, 0},
      {0, 2, 2},
      {0, 0, 3},
    };
    Vec<3> rhs{1, 10, 9};
    auto x = fix::solve<Vec<3>>(lhs, rhs);
    fmt::print("x: {}\n", x);
    THES_ALWAYS_ASSERT(x == vec3);
  }
  {
    Mat<2> lhs{{1, 0}, {2, 2}};
    Vec<2> rhs{1, 8};
    auto x = fix::solve<Vec<2>>(lhs, rhs);
    fmt::print("x: {}\n", x);
    THES_ALWAYS_ASSERT(x == vec2);
  }
  {
    Mat<1> lhs{{4}};
    Vec<1> rhs{12};
    auto x = fix::solve<Vec<1>>(lhs, rhs);
    fmt::print("x: {}\n", x);
    THES_ALWAYS_ASSERT(x == vec1);
  }
  fmt::print("\n");

  {
    using Tri = fix::DenseTriangularMatrix<thes::f64, 4, lineal::tri_lower>;
    constexpr std::array<std::pair<std::size_t, std::size_t>, 10> ref_pos{
      {{0, 0}, {1, 0}, {1, 1}, {2, 0}, {2, 1}, {2, 2}, {3, 0}, {3, 1}, {3, 2}, {3, 3}}};

    constexpr auto positions =
      thes::star::index_transform<Tri::element_num>([](auto i) { return Tri::index2pos(i); });
    fmt::print("tril positions: {}\n", positions);
    static_assert(thes::star::static_apply<std::max(decltype(positions)::size, ref_pos.size())>(
      [&]<std::size_t... tI>() { return (... && (get<tI>(positions) == get<tI>(ref_pos))); }));

    constexpr auto indices = thes::star::index_transform<Tri::element_num>([](auto i) {
      const auto [r, c] = Tri::index2pos(i);
      return Tri::pos2index(r, c);
    });
    fmt::print("tril indices: {}\n", indices);
    static_assert(thes::star::static_apply<decltype(indices)::size>(
      [&]<std::size_t... tI>() { return (... && (get<tI>(indices) == tI)); }));
    fmt::print("\n");
  }
  {
    using Tri = fix::DenseTriangularMatrix<thes::f64, 4, lineal::tri_upper>;
    constexpr std::array<std::pair<std::size_t, std::size_t>, 10> ref_pos{
      {{0, 0}, {0, 1}, {0, 2}, {0, 3}, {1, 1}, {1, 2}, {1, 3}, {2, 2}, {2, 3}, {3, 3}}};

    constexpr auto positions =
      thes::star::index_transform<Tri::element_num>([](auto i) { return Tri::index2pos(i); });
    fmt::print("triu positions: {}\n", positions);
    static_assert(thes::star::static_apply<std::max(decltype(positions)::size, ref_pos.size())>(
      [&]<std::size_t... tI>() { return (... && (get<tI>(positions) == get<tI>(ref_pos))); }));

    constexpr auto indices = thes::star::index_transform<Tri::element_num>([](auto i) {
      const auto [r, c] = Tri::index2pos(i);
      return Tri::pos2index(r, c);
    });
    fmt::print("triu indices: {}\n", indices);
    static_assert(thes::star::static_apply<decltype(indices)::size>(
      [&]<std::size_t... tI>() { return (... && (get<tI>(indices) == tI)); }));
    fmt::print("\n");
  }

  {
    fmt::print(thes::underline | thes::bold, "fixed 2×2 base LU\n");
    constexpr FixMatrixTester<double, 2> tester{
      .mat = Mat<2>{{2, 3}, {2, 4}},
      .inv = Mat<2>{{2, -1.5}, {-1, 1}},
      .rhs = Vec<2>{2, 3},
      .sol = Vec<2>{-0.5, 1},
    };
    test_base<tester>();
    test_lu<tester>();
    fmt::print("\n");
  }
  {
    fmt::print(thes::underline | thes::bold, "fixed 2×2 Cholesky\n");
    constexpr FixMatrixTester<double, 2> tester{
      .mat = Mat<2>{{2, 1}, {1, 13}},
      .inv = Mat<2>{{0.52, -0.04}, {-0.04, 0.08}},
      .rhs = Vec<2>{2, 3},
      .sol = Vec<2>{0.92, 0.16},
    };
    test_base<tester>();
    test_lu<tester>();
    test_cholesky<tester>();
    fmt::print("\n");
  }

  {
    fmt::print(thes::underline | thes::bold, "fixed 3×3 base\n");
    constexpr FixMatrixTester<double, 3> tester{
      .mat = Mat<3>{{1, 2, 3}, {1, 2, 4}, {2, 3, 5}},
      .inv = Mat<3>{{-2, -1, 2}, {3, -1, -1}, {-1, 1, 0}},
      .rhs = Vec<3>{2, 3, 4},
      .sol = Vec<3>{1, -1, 1},
    };
    test_base<tester>();
    fmt::print("\n");
  }
  {
    fmt::print(thes::underline | thes::bold, "fixed 3×3 LU\n");
    constexpr FixMatrixTester<double, 3> tester{
      .mat = Mat<3>{{1, 3, 2}, {1, 4, 2}, {2, 3, 5}},
      .inv = Mat<3>{{14, -9, -2}, {-1, 1, 0}, {-5, 3, 1}},
      .rhs = Vec<3>{2, 3, 4},
      .sol = Vec<3>{-7, 1, 3},
    };
    test_base<tester>();
    test_lu<tester>();
    fmt::print("\n");
  }
  {
    fmt::print(thes::underline | thes::bold, "fixed 3×3 Cholesky\n");
    constexpr FixMatrixTester<double, 3> tester{
      .mat = Mat<3>{{1, 1, 1}, {1, 3, 2}, {1, 2, 4}},
      .inv = Mat<3>{{1.6, -0.4, -0.2}, {-0.4, 0.6, -0.2}, {-0.2, -0.2, 0.4}},
      .rhs = Vec<3>{2, 3, 6},
      .sol = Vec<3>{0.8, -0.2, 1.4},
    };
    test_base<tester>();
    test_lu<tester>();
    test_cholesky<tester>();
    fmt::print("\n");
  }

  {
    fmt::print(thes::underline | thes::bold, "fixed 4×4 base\n");
    constexpr FixMatrixTester<double, 4> tester{
      .mat = Mat<4>{{1, 2, 3, 0}, {1, 2, 4, 0}, {2, 3, 5, 0}, {0, 0, 0, 2}},
      .inv = Mat<4>{{-2, -1, 2, 0}, {3, -1, -1, 0}, {-1, 1, 0, 0}, {0, 0, 0, 0.5}},
      .rhs = Vec<4>{2, 3, 4, 5},
      .sol = Vec<4>{1, -1, 1, 2.5},
    };
    test_base<tester>();
    fmt::print("\n");
  }
  {
    fmt::print(thes::underline | thes::bold, "fixed 4×4 LU\n");
    constexpr FixMatrixTester<double, 4> tester{
      .mat = Mat<4>{{1, 5, 2, 2}, {1, 6, 2, 1}, {2, 3, 7, 1}, {1, 1, 4, 1}},
      .inv =
        Mat<4>{
          {3.8, -4.2, 4.8, -8.2},
          {-0.4, 0.6, -0.4, 0.6},
          {-1, 1, -1, 2},
          {0.6, -0.4, -0.4, 0.6},
        },
      .rhs = Vec<4>{2, 3, 4, 5},
      .sol = Vec<4>{-26.8, 2.4, 7, 1.4},
    };
    test_base<tester>();
    test_lu<tester>();
    fmt::print("\n");
  }
}
