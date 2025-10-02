// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>

#include "thesauros/io.hpp"
#include "thesauros/ranges.hpp"
#include "thesauros/test.hpp"

#include "lineal/lineal.hpp"

#include "aux/default.hpp"

namespace test = lineal::test;

using Tlax = test::DefaultSharedDefs<>;

using Defs = Tlax::Defs;
using Real = Defs::Real;
using LoReal = Defs::LoReal;
using Size = Defs::Size;
using Ref = std::vector<LoReal>;

int main() {
  constexpr std::size_t vector_size = 4;
  constexpr Size size = 17;
  constexpr grex::PartTag<vector_size> vtag3{3};
  constexpr grex::FullTag<vector_size> vtag4{};
  using Vec = grex::Vector<LoReal, vector_size>;

  auto range_eq = [&](auto&& vec1, auto&& vec2) {
    THES_ALWAYS_ASSERT(thes::test::range_eq(vec1, vec2));
  };

  const auto expo = Tlax::make_expo(8U);

  lineal::DenseVector<LoReal, Size> vec(size);
  for (const auto i : thes::range(size)) {
    vec[i] = LoReal(i + 1);
  }

  auto grex_eq = [](const auto vec1, const auto vec2) {
    fmt::print("{}/{}\n", vec1, vec2);
    THES_ALWAYS_ASSERT(grex::horizontal_and(vec1 == vec2, grex::full_tag<vector_size>));
  };

  auto euclidean_squared = [&](const auto& vector) {
    const auto val1 = lineal::euclidean_squared<Real>(vector, expo);
    const auto val2 = std::transform_reduce(vec.begin(), vec.end(), 0.0, std::plus<>{},
                                            [](auto v) { return v * v; });
    fmt::print("euclidean_squared: {}/{}\n", thes::json_print(val1), thes::json_print(val2));
    THES_ALWAYS_ASSERT(val1 == val2);
  };
  auto euclidean = [&](const auto& vector) {
    const auto val1 = lineal::euclidean_norm<Real>(vector, expo);
    const auto val2 = std::sqrt(std::transform_reduce(vec.begin(), vec.end(), 0.0, std::plus<>{},
                                                      [](auto v) { return v * v; }));
    fmt::print("euclidean: {}/{}\n", thes::json_print(val1), thes::json_print(val2));
    THES_ALWAYS_ASSERT(val1 == val2);
  };
  auto inv_euclidean = [&](const auto& vector) {
    const auto val1 = lineal::inv_euclidean_norm<Real>(vector, expo);
    const auto val2 =
      1.0 / std::sqrt(std::transform_reduce(vec.begin(), vec.end(), 0.0, std::plus<>{},
                                            [](auto v) { return v * v; }));
    fmt::print("inv_euclidean: {}/{}\n", thes::json_print(val1), thes::json_print(val2));
    THES_ALWAYS_ASSERT(val1 == val2);
  };
  auto dot = [&](const auto& vec1, const auto& vec2) {
    const auto val1 = lineal::dot<Real>(vec1, vec2, expo);
    const auto val2 =
      std::transform_reduce(vec1.begin(), vec1.end(), vec2.begin(), 0.0, std::plus<>{},
                            [](auto v1, auto v2) { return Real(v1) * v2; });
    fmt::print("dot: {}/{}\n", thes::json_print(val1), thes::json_print(val2));
    THES_ALWAYS_ASSERT(thes::pow<2>(val1 - val2) < std::numeric_limits<Real>::epsilon());
  };

  {
    Ref ref_one(size);
    Ref ref_vec(size);
    Ref ref_dbl(size);
    Ref ref_tpl(size);
    Ref ref_sq(size);
    for (const auto i : thes::range(size)) {
      ref_one[i] = 1;
      const auto v = LoReal(i + 1);
      ref_vec[i] = v;
      ref_dbl[i] = 2 * v;
      ref_tpl[i] = 3 * v;
      ref_sq[i] = v * v;
    }

    range_eq(vec, ref_vec);
    range_eq(vec + vec, ref_dbl);
    range_eq(vec - LoReal{1} * vec, Ref(size, 0));
    // range_eq(lineal::cw_multiply<LoReal>(vec, vec), ref_sq);
    // range_eq(lineal::cw_divide<LoReal>(vec, vec), Ref(size, 1));
    range_eq(vec * LoReal{2}, ref_dbl);
    range_eq(LoReal{3} * vec, ref_tpl);
  }

  range_eq(vec - vec, lineal::constant(size, LoReal{0}));
  // range_eq(lineal::cw_divide<LoReal>(vec, vec), lineal::constant(size, LoReal{1}));

  grex_eq(vec.compute(size - 3, vtag3), Vec{15, 16, 17, 0});
  grex_eq(vec.compute(size - 3, vtag3), (vec.begin() + (size - 3)).compute(vtag3));
  grex_eq(vec.compute(Size{4}, vtag4), Vec(5, 6, 7, 8));
  grex_eq(vec.compute(Size{4}, vtag4), (vec.begin() + 4).compute(vtag4));

  euclidean_squared(vec);
  euclidean(vec);
  inv_euclidean(vec);
  dot(vec, 1.2 * vec);

  lineal::DenseVector<LoReal, Size> stto1(size);
  lineal::assign(stto1, vec, expo);
  range_eq(stto1, vec);
  range_eq(lineal::assign_expr(stto1, 2.0 * vec), 2.0 * vec);
  range_eq(stto1, vec * LoReal{2});
  range_eq(lineal::create_from<lineal::DenseVector<Real, Size>>(vec - vec, expo), vec * 0.0);

  range_eq(sub_vector(thes::range<Size>(4), vec), Ref{1, 2, 3, 4});
  range_eq(sub_vector(thes::range<Size>(4, 8), vec), Ref{5, 6, 7, 8});

  lineal::swap(stto1, vec, expo);
  range_eq(vec, stto1 * LoReal{2});
}
