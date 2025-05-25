// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef TEST_SHARED_STENCIL_HPP
#define TEST_SHARED_STENCIL_HPP

#include <array>
#include <concepts>
#include <functional>
#include <type_traits>
#include <vector>

#include "thesauros/ranges.hpp"
#include "thesauros/static-ranges.hpp"
#include "thesauros/test.hpp"
#include "thesauros/types.hpp"
#include "thesauros/utility.hpp"

#include "lineal/lineal.hpp"

#include "aux/default.hpp"
#include "aux/iterative.hpp"
#include "aux/materials.hpp"
#include "aux/msg.hpp"
#include "aux/test.hpp"

namespace test {
enum struct LinearSystemKind : thes::u8 {
  LOOKUP_STENCIL,
  VALUE_STENCIL,
  FULL_CSR,
  COMPRESSED_CSR,
};

template<LinearSystemKind tKind>
inline void run() {
  namespace msg = lineal::msg;
  namespace test = lineal::test;

  using Tlax = test::DefaultSharedDefs;

  using Defs = Tlax::Defs;
  using Real = Defs::Real;
  using LoReal = Defs::LoReal;
  using Size = Defs::Size;
  using SysInfo = Tlax::LookupSymmetricDiffInfo;

  constexpr auto csr_map = thes::StaticMap{
    thes::static_kv<LinearSystemKind::FULL_CSR, true>,
    thes::static_kv<LinearSystemKind::COMPRESSED_CSR, true>,
  };
  constexpr bool is_csr = csr_map.get(thes::auto_tag<tKind>, false);

  constexpr auto valuator_map = thes::StaticMap{
    thes::static_kv<LinearSystemKind::LOOKUP_STENCIL, thes::type_tag<Tlax::LookupValuator>>,
    thes::static_kv<LinearSystemKind::VALUE_STENCIL, thes::type_tag<Tlax::ValueValuator>>,
    thes::static_kv<LinearSystemKind::FULL_CSR, thes::type_tag<Tlax::LookupValuator>>,
    thes::static_kv<LinearSystemKind::COMPRESSED_CSR, thes::type_tag<Tlax::LookupValuatorZero>>,
  };
  using Valuator = std::decay_t<decltype(valuator_map.get(thes::auto_tag<tKind>))>::Type;

  using FineLhs = std::conditional_t<is_csr, Tlax::CsrMatrix<Real>, Tlax::StencilMatrix<Valuator>>;
  using FineRhs =
    std::conditional_t<is_csr, Tlax::DenseVector<Real>, Tlax::StencilVector<Valuator>>;
  using FineVec = Tlax::DenseVector<Real>;

  using CoarseLhs = Tlax::CsrMatrix<LoReal>;
  using CoarseVec = Tlax::DenseVector<Real>;

  constexpr SysInfo::SizeArray dims{4, 9, 7};
  constexpr SysInfo::MaterialCoeffs coeffs{0.0, 1.0, 2.0, 3.0};
  constexpr SysInfo::NoneMaterialMap none_mats{false, false, false, false};

  SysInfo sys_info{
    dims,
    dims | thes::star::transform([](Size idx) { return Real(idx); }) | thes::star::to_array,
    coeffs,
    none_mats,
    /*solution_start=*/1,
    /*solution_end=*/0,
  };
  THES_ALWAYS_ASSERT((dims | thes::star::left_reduce(std::multiplies{})) == sys_info.total_size());
  THES_ALWAYS_ASSERT(test::star_eq(sys_info.axis_lengths(), dims));
  THES_ALWAYS_ASSERT(test::star_eq(sys_info.sizes(), dims));
  THES_ALWAYS_ASSERT(
    test::star_eq(sys_info.from_sizes(),
                  std::array<Size, 4>{dims[0] * dims[1] * dims[2], dims[1] * dims[2], dims[2], 1}));
  THES_ALWAYS_ASSERT(test::star_eq(sys_info.axis_quotients(), thes::star::constant<3>(1)));

  const auto env = Tlax::make_env(2U);
  decltype(auto) expo = env.execution_policy();

  auto material =
    lineal::materials::diagonal_lines<Defs>(sys_info, thes::range(sys_info.total_size()), expo);
  const Valuator valuator{sys_info, material};
  const auto sys = [&] {
    if constexpr (is_csr) {
      return lineal::csr_from_adjacent_stencil<FineLhs, FineRhs>(valuator, env);
    } else {
      return lineal::System{
        .lhs = FineLhs{valuator},
        .rhs = FineRhs{valuator},
        .compress_map = thes::Empty{},
      };
    }
  }();

  if constexpr (std::same_as<decltype(sys.compress_map), thes::Empty>) {
    THES_ALWAYS_ASSERT(sys.lhs.row_num() == sys_info.total_size() &&
                       sys.lhs.column_num() == sys_info.total_size());
  } else {
    THES_ALWAYS_ASSERT(sys.lhs.row_num() == sys.lhs.column_num());
  }

  FineVec factor_vec{sys.rhs.size(), 2, env};
  const auto prod = sys.lhs * factor_vec;
  THES_ALWAYS_ASSERT(test::vector_eq(prod, sys.lhs * lineal::constant(sys.rhs.size(), Real{2})));

  const auto prod_vec = lineal::create_from<FineVec>(prod, expo);
  THES_ALWAYS_ASSERT(test::vector_eq(prod, prod_vec, 1e-14));

  const auto wright_instance = Tlax::make_wright<FineVec, CoarseLhs, CoarseVec>();
  THES_ALWAYS_ASSERT((test::chol<Real, CoarseVec>(sys.lhs, wright_instance, env) <= 1e-13));

  {
    FineVec sol(sys.rhs.size(), 0, env);

    std::vector<Real> residuals{1e-12, 1e-8, 1e-13, 1e-13, 1e-13, 1e-13, 1e-13};
    auto it = residuals.begin();

    auto op = [&](auto res, auto& /*sol*/, const auto& envi) {
      const auto bound = *(it++);
      const auto err_max = lineal::max_norm<Real>(sol - factor_vec, expo);
      envi.log("error", msg::ResidualErrorBound{res, err_max, bound});
      THES_ALWAYS_ASSERT(res <= bound && err_max <= bound);
    };
    test::suite<Defs>(sys.lhs, sol, prod_vec, wright_instance, env, op);
  }

  {
    FineVec vec(sys.rhs.size(), env);
    assign(vec, sys.rhs, expo);
    THES_ALWAYS_ASSERT(test::vector_eq(vec, sys.rhs));
  }

  {
    FineVec sol(sys.rhs.size(), 0, env);

    std::vector<Real> residuals{1e-13, 1e-9, 1e-14, 1e-14, 1e-14, 1e-13, 2e-14};
    auto it = residuals.begin();
    const auto ref = lineal::materials::diagonal_lines_eff_diff<Defs>(sys_info);

    auto op = [&, effdiff = lineal::EffDiffCalc{valuator, sys.compress_map, env}](
                auto res, auto& /*sol*/, const auto& envi) {
      using namespace thes::literals;

      const auto bound = *(it++);
      const auto err_max = lineal::max_norm<Real>(sys.lhs * sol - sys.rhs, expo);
      envi.log("error", msg::ResidualsBound{res, err_max, bound});
      THES_ALWAYS_ASSERT(res <= bound && err_max <= bound);

      const auto eff_diff = effdiff(valuator, sys.compress_map, sol, envi);
      envi.log("eff_diff", msg::RefValue{eff_diff, ref});
      THES_ALWAYS_ASSERT(thes::is_close<Real>(eff_diff, ref, "abs_tol"_key = bound));
    };
    test::suite<Defs>(sys.lhs, sol, sys.rhs, wright_instance, env, op);
  }
}
} // namespace test

#endif // TEST_SHARED_STENCIL_HPP
