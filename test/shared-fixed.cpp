// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <cstddef>

#include "thesauros/format.hpp"
#include "thesauros/test.hpp"

#include "lineal/lineal.hpp"

template<std::size_t tDim>
using Mat = lineal::fix::DenseMatrix<double, tDim, tDim>;
template<std::size_t tDim>
using Vec = lineal::fix::DenseVector<double, tDim>;

static constexpr Mat<3> mat3x3{
  {1, 0, 0},
  {0, 2, 2},
  {0, 0, 3},
};
static constexpr Vec<3> vec3{1, 2, 3};
static constexpr auto prod3 = mat3x3 * vec3;
static constexpr Vec<3> ref3{1, 10, 9};
static_assert(Vec<3>{prod3} == ref3);
static constexpr auto sol3 = lineal::fix::solve<Vec<3>>(mat3x3, ref3);
static_assert(sol3 == vec3);

static constexpr Mat<2> mat2x2{{1, 0}, {2, 2}};
static constexpr Vec<2> vec2{1, 3};
static constexpr auto prod2 = mat2x2 * vec2;
static constexpr Vec<2> ref2{1, 8};
static_assert(Vec<2>{prod2} == ref2);
static constexpr auto sol2 = lineal::fix::solve<Vec<2>>(mat2x2, ref2);
static_assert(sol2 == vec2);

static constexpr Mat<1> mat1x1{{4}};
static constexpr Vec<1> vec1{3};
static constexpr auto prod1 = mat1x1 * vec1;
static constexpr Vec<1> ref1{12};
static_assert(Vec<1>{prod1} == ref1);
static constexpr auto sol1 = lineal::fix::solve<Vec<1>>(mat1x1, ref1);
static_assert(sol1 == vec1);

int main() {
  {
    Mat<3> lhs{
      {1, 0, 0},
      {0, 2, 2},
      {0, 0, 3},
    };
    Vec<3> rhs{1, 10, 9};
    auto x = lineal::fix::solve<Vec<3>>(lhs, rhs);
    fmt::print("x: {}\n", x);
    THES_ALWAYS_ASSERT(x == vec3);
  }
  {
    Mat<2> lhs{{1, 0}, {2, 2}};
    Vec<2> rhs{1, 8};
    auto x = lineal::fix::solve<Vec<2>>(lhs, rhs);
    fmt::print("x: {}\n", x);
    THES_ALWAYS_ASSERT(x == vec2);
  }
  {
    Mat<1> lhs{{4}};
    Vec<1> rhs{12};
    auto x = lineal::fix::solve<Vec<1>>(lhs, rhs);
    fmt::print("x: {}\n", x);
    THES_ALWAYS_ASSERT(x == vec1);
  }
}
