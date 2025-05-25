// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_TENSOR_FIXED_SOLVE_HPP
#define INCLUDE_LINEAL_TENSOR_FIXED_SOLVE_HPP

#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <type_traits>

#include "stado/stado.hpp"
#include "thesauros/macropolis.hpp"
#include "thesauros/static-ranges.hpp"
#include "thesauros/types.hpp"

#include "lineal/tensor/fixed/concepts.hpp"
#include "lineal/vectorization.hpp"

namespace lineal::fix {
template<AnyMatrix TLhs, AnyVector TRhs>
requires(TLhs::dimensions.row_num == TLhs::dimensions.column_num &&
         TLhs::dimensions.row_num == TRhs::size)
THES_ALWAYS_INLINE constexpr void gauss_elimination(TLhs& lhs, TRhs& rhs) {
  using LVal = TLhs::Value;
  constexpr std::size_t size = TRhs::size;

  thes::star::iota<0, size> | thes::star::for_each([&](auto k) {
    // Partial Pivoting
    std::size_t imax = k;
    LVal lmax = lhs(k, k);
    thes::star::iota<k + 1, size> | thes::star::for_each([&](auto i) {
      if (const auto v = std::abs(lhs(i, k)); v > lmax) {
        imax = i;
        lmax = v;
      }
    });
    thes::star::iota<0, size> |
      thes::star::for_each([&](auto j) { std::swap(lhs(imax, j), lhs(k, j)); });
    std::swap(rhs[imax], rhs[k]);

    // Make the elements below the pivot elements equal to zero
    thes::star::iota<k + 1, size> | thes::star::for_each([&](auto i) {
      auto factor = lhs(i, k) / lhs(k, k);
      thes::star::iota<k, size> |
        thes::star::for_each([&](auto j) { lhs(i, j) -= factor * lhs(k, j); });
      rhs[i] -= factor * rhs[k];
    });
  });
}

template<AnyVector TSol, AnyMatrix TLhs, AnyVector TRhs>
requires(TLhs::dimensions.row_num == TLhs::dimensions.column_num &&
         TLhs::dimensions.row_num == TRhs::size && TSol::size == TRhs::size)
THES_ALWAYS_INLINE constexpr TSol backward_substitution(const TLhs& lhs, const TRhs& rhs) {
  using Real = TSol::Value;
  constexpr std::size_t size = TRhs::size;

  TSol x{};
  thes::star::iota<0, size> | thes::star::reversed | thes::star::for_each([&](auto i) {
    Real tmp = rhs[i];
    thes::star::iota<i + 1, size> | thes::star::for_each([&](auto j) { tmp -= lhs(i, j) * x[j]; });
    x[i] = tmp / lhs(i, i);
  });
  return x;
}

template<AnyVector TSol, AnyMatrix TLhs, AnyVector TRhs>
requires(TLhs::dimensions.row_num == TLhs::dimensions.column_num &&
         TLhs::dimensions.row_num == TRhs::size && TSol::size == TRhs::size)
THES_ALWAYS_INLINE constexpr TSol solve(const TLhs& lhs, const TRhs& rhs) {
  constexpr std::size_t size = TRhs::size;
  if constexpr (size == 1) {
    return TSol{rhs[thes::index_tag<0>] / lhs(thes::index_tag<0>, thes::index_tag<0>)};
  }
  if constexpr (size == 2) {
    const auto a = lhs(thes::index_tag<0>, thes::index_tag<0>);
    const auto b = lhs(thes::index_tag<0>, thes::index_tag<1>);
    const auto c = lhs(thes::index_tag<1>, thes::index_tag<0>);
    const auto d = lhs(thes::index_tag<1>, thes::index_tag<1>);

    const auto b0 = rhs[thes::index_tag<0>];
    const auto b1 = rhs[thes::index_tag<1>];

    const auto det = a * d - b * c;

    if (!std::is_constant_evaluated()) {
      using Real = TSol::Value;
      using Vec2 = stado::Vector<Real, 2>;

      if constexpr (std::same_as<Real, double>) {
        const Vec2 rhs_vec = grex::load_ptr(rhs.data(), grex::VectorSize<2>{});
        const Vec2 rhs_rev = stado::permute2<1, 0>(rhs_vec);
        const Vec2 out = stado::mul_sub(rhs_vec, Vec2{d, a}, rhs_rev * Vec2{b, c}) / det;

        return TSol{
          out.template extract<0>(),
          out.template extract<1>(),
        };
      }
    }
    return TSol{
      (d * b0 - b * b1) / det,
      (a * b1 - c * b0) / det,
    };
  }
  if constexpr (size == 3) {
    const auto a = lhs(thes::index_tag<0>, thes::index_tag<0>);
    const auto b = lhs(thes::index_tag<0>, thes::index_tag<1>);
    const auto c = lhs(thes::index_tag<0>, thes::index_tag<2>);
    const auto d = lhs(thes::index_tag<1>, thes::index_tag<0>);
    const auto e = lhs(thes::index_tag<1>, thes::index_tag<1>);
    const auto f = lhs(thes::index_tag<1>, thes::index_tag<2>);
    const auto g = lhs(thes::index_tag<2>, thes::index_tag<0>);
    const auto h = lhs(thes::index_tag<2>, thes::index_tag<1>);
    const auto i = lhs(thes::index_tag<2>, thes::index_tag<2>);

    const auto b0 = rhs[thes::index_tag<0>];
    const auto b1 = rhs[thes::index_tag<1>];
    const auto b2 = rhs[thes::index_tag<2>];

    if (!std::is_constant_evaluated()) {
      using Real = TSol::Value;
      using Vec4 = stado::Vector<Real, 4>;

      if constexpr (std::same_as<Real, thes::f32> ||
                    (STADO_INSTRUCTION_SET >= STADO_AVX2 && std::same_as<Real, thes::f64>)) {
        Vec4 abc{a, b, c, d};
        Vec4 def{d, e, f, g};
        Vec4 ghi{g, h, i, 0};

        Vec4 bca = stado::permute4<1, 2, 0, 3>(abc);
        Vec4 cab = stado::permute4<2, 0, 1, 3>(abc);
        Vec4 efd = stado::permute4<1, 2, 0, 3>(def);
        Vec4 fde = stado::permute4<2, 0, 1, 3>(def);
        Vec4 hig = stado::permute4<1, 2, 0, 3>(ghi);
        Vec4 igh = stado::permute4<2, 0, 1, 3>(ghi);

        // Because ghi[3] == 0, both (efd * igh)[3] == 0 and (fde * hig)[3] == 0
        Vec4 c0 = stado::mul_sub(efd, igh, fde * hig);
        Vec4 c1 = stado::mul_sub(cab, hig, bca * igh);
        Vec4 c2 = stado::mul_sub(bca, fde, cab * efd);

        auto det = abc * c0;
        det += stado::permute4<1, 0, 3, 2>(det);
        det += stado::permute4<2, 3, 0, 1>(det);

        auto out = stado::mul_add(c2, b2, stado::mul_add(c1, b1, c0 * b0)) / det;
        return TSol{
          out.template extract<0>(),
          out.template extract<1>(),
          out.template extract<2>(),
        };
      }
    }

    const auto inv00 = e * i - f * h;
    const auto inv10 = f * g - d * i;
    const auto inv20 = d * h - e * g;

    const auto inv01 = c * h - b * i;
    const auto inv11 = a * i - c * g;
    const auto inv21 = b * g - a * h;

    const auto inv02 = b * f - c * e;
    const auto inv12 = c * d - a * f;
    const auto inv22 = a * e - b * d;

    const auto det = a * inv00 + b * inv10 + c * inv20;

    return TSol{
      (inv00 * b0 + inv01 * b1 + inv02 * b2) / det,
      (inv10 * b0 + inv11 * b1 + inv12 * b2) / det,
      (inv20 * b0 + inv21 * b1 + inv22 * b2) / det,
    };
  }

  TLhs lhsc{lhs};
  TRhs rhsc{rhs};
  gauss_elimination(lhsc, rhsc);
  return backward_substitution<TSol>(lhsc, rhsc);
}
} // namespace lineal::fix

#endif // INCLUDE_LINEAL_TENSOR_FIXED_SOLVE_HPP
