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
#include <tuple>
#include <type_traits>
#include <utility>
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
enum struct LinearSystemKind : thes::u8 { lookup_stencil, value_stencil, full_csr, compressed_csr };

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
    thes::static_kv<LinearSystemKind::full_csr, true>,
    thes::static_kv<LinearSystemKind::compressed_csr, true>,
  };
  constexpr bool is_csr = csr_map.get(thes::auto_tag<tKind>, false);

  constexpr auto valuator_map = thes::StaticMap{
    thes::static_kv<LinearSystemKind::lookup_stencil, thes::type_tag<Tlax::LookupValuator>>,
    thes::static_kv<LinearSystemKind::value_stencil, thes::type_tag<Tlax::ValueValuator>>,
    thes::static_kv<LinearSystemKind::full_csr, thes::type_tag<Tlax::LookupValuator>>,
    thes::static_kv<LinearSystemKind::compressed_csr, thes::type_tag<Tlax::LookupValuatorZero>>,
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

  const auto envvc = Tlax::make_env(2U);
  const auto envsc = lineal::Environment{
    std::in_place, [&] { return Tlax::make_expo(2U, thes::index_tag<1>); }, envvc.logger()};
  decltype(auto) expovc = envvc.execution_policy();
  decltype(auto) exposc = envsc.execution_policy();

  const auto& env = envvc;
  const auto& expo = expovc;

  auto materialvc =
    lineal::materials::diagonal_lines<Defs>(sys_info, thes::range(sys_info.total_size()), expovc);
  auto materialsc =
    lineal::materials::diagonal_lines<Defs>(sys_info, thes::range(sys_info.total_size()), exposc);
  THES_ALWAYS_ASSERT(thes::test::range_eq(materialvc, materialsc));
  const Valuator valuator{sys_info, materialsc};
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

  FineVec factorvc{sys.rhs.size(), 2, envvc};
  const auto prod = sys.lhs * factorvc;
  THES_ALWAYS_ASSERT(test::vector_eq(prod, sys.lhs * lineal::constant(sys.rhs.size(), Real{2})));

  const auto prodvc = lineal::create_from<FineVec>(prod, expovc);
  const auto prodsc = lineal::create_from<FineVec>(prod, exposc);
  THES_ALWAYS_ASSERT(test::vector_eq(prodvc, prodsc, 1e-14));
  THES_ALWAYS_ASSERT(test::vector_eq(prod, prodvc, 1e-14));

  // TODO find a way to use sub-environments even with this messy setup
  auto solver_tests = [&](thes::TypedValueTag<lineal::IsSymmetric> auto symmetric) {
    const auto wright_instance = Tlax::make_wright<FineVec, CoarseLhs, CoarseVec, symmetric>();
    env.log("coarse_solver", thes::type_name(wright_instance.coarse_solver()));
    THES_ALWAYS_ASSERT((test::direct<Real, CoarseVec>(sys.lhs, wright_instance, env) <= 1e-13));

    if (envvc.execution_policy().thread_num() == 1) {
      lineal::SorSolver<Real, void, lineal::forward_tag, lineal::SorVariant::regular> solver{0.9};

      FineVec solvc(sys.rhs.size(), 0, envvc);
      FineVec auxvc(sys.rhs.size(), 0, envvc);
      auto instvc = solver.instantiate(sys.lhs, envvc);
      instvc.apply(solvc, sys.rhs, std::tie(auxvc), envvc);

      FineVec solsc(sys.rhs.size(), 0, envsc);
      FineVec auxsc(sys.rhs.size(), 0, envvc);
      auto instsc = solver.instantiate(sys.lhs, envsc);
      instsc.apply(solsc, sys.rhs, std::tie(auxsc), envsc);
      THES_ALWAYS_ASSERT(test::vector_eq(solvc, solsc, 1e-14));
    }

    {
      FineVec sol(sys.rhs.size(), 0, env);

      std::vector<Real> residuals{
        // regular
        /*cg=*/1e-12,
        /*bicgstab=*/1e-12,
        /*sor=*/1e-8,
        /*ssor1=*/1e-13,
        /*sor5=*/1e-13,
        /*ssor5=*/1e-13,
        /*pcg_ssor1=*/1e-13,
        /*pbicgstab_ssor1=*/1e-12,
        /*pcg_amg=*/1e-13,
        /*pbicgstab_amg=*/1e-13,
        // ultra
        /*cg=*/1e-12,
        /*bicgstab=*/1e-12,
        /*sor=*/1e-8,
        /*ssor1=*/1e-13,
        /*sor5=*/1e-13,
        /*ssor5=*/1e-13,
        /*pcg_ssor1=*/1e-13,
        /*pbicgstab_ssor1=*/1e-12,
        /*pcg_amg=*/1e-13,
        /*pbicgstab_amg=*/1e-13,
      };
      auto it = residuals.begin();

      auto op = [&](auto res, auto& /*sol*/, const auto& envi) {
        const auto bound = *(it++);
        const auto err_max = lineal::max_norm<Real>(sol - factorvc, expo);
        envi.log("error", msg::ResidualErrorBound{res, err_max, bound});
        THES_ALWAYS_ASSERT(res <= bound && err_max <= bound);
      };
      test::suite<Defs>(sys.lhs, sol, prodvc, wright_instance, env, op);
    }

    {
      FineVec vec(sys.rhs.size(), env);
      lineal::assign(vec, sys.rhs, expo);
      THES_ALWAYS_ASSERT(test::vector_eq(vec, sys.rhs));
    }

    {
      FineVec sol(sys.rhs.size(), 0, env);

      std::vector<Real> residuals{
        // regular
        /*cg=*/1e-13,
        /*bicgstab=*/1e-13,
        /*sor=*/1e-9,
        /*ssor1=*/1.1e-14,
        /*sor5=*/1e-14,
        /*ssor5=*/1.1e-14,
        /*pcg_ssor1=*/1e-13,
        /*pbicgstab_ssor1=*/1e-13,
        /*pcg_amg=*/2e-14,
        /*pbicgstab_amg=*/1e-13,
        // ultra
        /*cg=*/1e-13,
        /*bicgstab=*/1e-13,
        /*sor=*/1e-9,
        /*ssor1=*/1.1e-14,
        /*sor5=*/1e-14,
        /*ssor5=*/1.1e-14,
        /*pcg_ssor1=*/1e-13,
        /*pbicgstab_ssor1=*/1e-13,
        /*pcg_amg=*/2e-14,
        /*pbicgstab_amg=*/1e-13,
      };
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
  };
  solver_tests(thes::auto_tag<lineal::IsSymmetric{true}>);
  solver_tests(thes::auto_tag<lineal::IsSymmetric{false}>);
}
} // namespace test

#endif // TEST_SHARED_STENCIL_HPP
