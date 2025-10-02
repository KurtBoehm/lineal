// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_LINEAR_SOLVER_ITERATIVE_STATIONARY_JACOBI_SWEEP_HPP
#define INCLUDE_LINEAL_LINEAR_SOLVER_ITERATIVE_STATIONARY_JACOBI_SWEEP_HPP

#include <cassert>
#include <cmath>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>

#include "thesauros/macropolis.hpp"
#include "thesauros/static-ranges.hpp"
#include "thesauros/types.hpp"
#include "thesauros/utility.hpp"

#include "lineal/base.hpp"
#include "lineal/component-wise/exec-constraint.hpp"
#include "lineal/component-wise/facade.hpp"
#include "lineal/parallel.hpp"
#include "lineal/vectorization.hpp"

namespace lineal::detail {
struct JacobiSweepConf {
  using Work = void;
  using Value = void;
  static constexpr bool supports_const_access = false;
  static constexpr bool supports_mutable_access = true;
  static constexpr auto component_wise_seq = thes::Tuple{thes::index_tag<0>, thes::index_tag<3>};
};

template<typename TDistInfo, typename TReal, typename TThreadInfo, SharedMatrix TLhs,
         SharedVector TSolIn, SharedVector TSolOut, SharedVector TRhs>
struct JacobiSweepSink
    : public facades::ComponentWiseOp<
        JacobiSweepSink<TDistInfo, TReal, TThreadInfo, TLhs, TSolIn, TSolOut, TRhs>,
        JacobiSweepConf, TLhs, TSolIn, TSolOut, TRhs> {
  using Real = TReal;
  using Value = Real;
  using Lhs = std::decay_t<TLhs>;
  using Rhs = std::decay_t<TRhs>;
  using LhsValue = Lhs::Value;
  using InValue = std::decay_t<TSolIn>::Value;
  using OutValue = std::decay_t<TSolOut>::Value;
  static constexpr bool is_shared = std::is_void_v<TDistInfo>;

  using Parent =
    facades::ComponentWiseOp<JacobiSweepSink, JacobiSweepConf, TLhs, TSolIn, TSolOut, TRhs>;
  using DistInfoStorage = thes::VoidStorage<TDistInfo>;

  using Size = SizeIntersection<Lhs, Rhs>;
  using LocalIdx = OptLocalIndex<is_shared, Size>;

  explicit JacobiSweepSink(DistInfoStorage&& dist_info, TThreadInfo&& thread_info, Real relax,
                           TLhs&& lhs, TSolIn&& sol_in, TSolOut&& sol_out, TRhs&& rhs)
      : Parent(std::forward<TLhs>(lhs), std::forward<TSolIn>(sol_in),
               std::forward<TSolOut>(sol_out), std::forward<TRhs>(rhs)),
        dist_info_(std::forward<DistInfoStorage>(dist_info)),
        thread_info_(std::forward<TThreadInfo>(thread_info)), relax_(relax) {}

  THES_ALWAYS_INLINE void compute_base_scalar(auto lhs_row, auto rhs_val) {
    const TSolIn& x_in = sol_in();
    TSolOut& x_out = sol_out();
    Real accum = rhs_val;

    auto update = [&](LocalIdx j, Real aij)
                    THES_ALWAYS_INLINE { accum = grex::fnmadd(aij, Real(x_in[j]), accum); };
    Real diag = std::numeric_limits<Real>::signaling_NaN();

    lhs_row.iterate(
      // j < index
      [&](LocalIdx j, Real aij) THES_ALWAYS_INLINE { update(j, aij); },
      // j == index
      [&]([[maybe_unused]] LocalIdx j, Real aij) THES_ALWAYS_INLINE { diag = aij; },
      // j > index
      [&](LocalIdx j, Real aij) THES_ALWAYS_INLINE { update(j, aij); }, valued_tag, unordered_tag);

    const LocalIdx i = lhs_row.ext_index();
    assert(std::isfinite(diag));
    const Real quot = accum / diag;
    x_out[i] = OutValue(grex::fmadd(relax_, quot, m1_relax_ * Real(x_in[i])));
  }

  template<grex::FullVectorTag TVecTag>
  THES_ALWAYS_INLINE void compute_base_multicolumn(auto row, auto rhs_val, TVecTag ftag) {
    const LocalIdx i = row.ext_index();
    const auto ivec = grex::broadcast(index_value(i), ftag);

    const TSolIn& x_in = sol_in();
    TSolOut& x_out = sol_out();
    const Real x_diag(x_in[i]);

    const Real rrhs(rhs_val);
    auto accum_vec = grex::broadcast(rrhs / ftag.size, ftag);
    auto dia_vec = grex::zeros(ftag);
    row.multicolumn_iterate(
      [&](auto j, auto aij, auto vtag) THES_ALWAYS_INLINE {
        const auto x_vec = grex::convert_unsafe<Real>(x_in.lookup(j, vtag));
        const auto aij_real = grex::convert_unsafe<Real>(aij);
        accum_vec = grex::fnmadd(aij_real, x_vec, accum_vec);
        dia_vec = grex::blend(vtag.mask(j == ivec), dia_vec, aij_real);
      },
      valued_tag, unordered_tag, ftag);

    const Real accum = grex::horizontal_add(accum_vec, ftag);
    const Real diag = grex::horizontal_add(dia_vec, ftag);
    assert(diag != 0);
    const Real quot = accum / diag;
    x_out[i] = OutValue(grex::fmadd(relax_, quot, m1_relax_ * x_diag));
  }

  template<grex::AnyTag TVecTag>
  THES_ALWAYS_INLINE void compute_base_simd(auto row, auto rhs_vec, TVecTag tag) {
    using Vec = grex::Vector<Real, tag.size>;

    const TSolIn& x_in = sol_in();
    auto it_in = x_in.iter_at(row.index());
    TSolOut& x_out = sol_out();

    const LocalIdx i = row.ext_index();
    auto accum = grex::convert_unsafe<Real>(rhs_vec);

#ifdef FINITE_CHECK
#error "FINITE_CHECK is already defined!"
#endif
#define FINITE_CHECK(x) assert(grex::horizontal_and(grex::is_finite(x), tag))

    auto update = [&](auto aij, auto vec_vec) THES_ALWAYS_INLINE {
      FINITE_CHECK(aij);
      FINITE_CHECK(vec_vec);
      accum = grex::fnmadd(aij, grex::convert_unsafe<Real>(vec_vec), accum);
      FINITE_CHECK(accum);
    };

    Vec diag{};
    Vec nb{};

    // TODO The dimension parameter is inconsistent with the interface described in the paper!
    row.banded_iterate(
      // j < index
      [&](auto aij, auto off, thes::AnyIndexTag auto /*dim*/)
        THES_ALWAYS_INLINE { update(aij, off(it_in).compute(tag)); },
      // j == index
      [&](auto aij, auto /*off*/) THES_ALWAYS_INLINE { diag = aij; },
      // j > index
      [&](auto aij, auto off, thes::AnyIndexTag auto /*dim*/)
        THES_ALWAYS_INLINE { update(aij, off(it_in).compute(tag)); },
      unordered_tag, tag);

    const Vec quot = accum / diag;
    const auto x_diag = grex::convert_unsafe<Real>(x_in.compute(i, tag));
    const auto x = grex::fmadd(Vec{relax_}, quot, m1_relax_ * x_diag);
    x_out.store(i, grex::convert_unsafe<OutValue>(x), tag);

#undef FINITE_CHECK
  }

  template<grex::AnyTag TTag>
  THES_ALWAYS_INLINE void compute_impl(auto lhs_off, auto rhs_off, auto rhs_load, TTag tag) {
    if constexpr ((grex::AnyScalarTag<TTag> ||
                   (grex::AnyVectorTag<TTag> && grex::is_geometry_respecting<TTag>)) &&
                  BandedIterable<Lhs, TTag>) {
      compute_base_simd(lhs_off(0), rhs_load(), tag);
    } else {
      grex::for_each<Size>(
        [&](auto i) THES_ALWAYS_INLINE { compute_base_scalar(lhs_off(i), rhs_off(i)); }, tag);
    }
  }

  THES_ALWAYS_INLINE void compute_iter(grex::AnyTag auto tag, const auto& /*children*/, auto lhs_it,
                                       auto rhs_it) {
    compute_impl([&](auto i) THES_ALWAYS_INLINE { return lhs_it[i]; },
                 [&](auto i) THES_ALWAYS_INLINE { return rhs_it[i]; },
                 [&]() THES_ALWAYS_INLINE { return rhs_it.compute(tag); }, tag);
  }
  THES_ALWAYS_INLINE void compute_base(grex::AnyTag auto tag, const auto& arg,
                                       const auto& /*children*/, const auto& lhs, const auto& rhs) {
    compute_impl([&](Size i) THES_ALWAYS_INLINE { return lhs[arg + i]; },
                 [&](Size i) THES_ALWAYS_INLINE { return rhs[arg + i]; },
                 [&]() THES_ALWAYS_INLINE { return rhs.compute(arg, tag); }, tag);
  }

private:
  [[nodiscard]] const TSolIn& sol_in() const {
    return thes::star::get_at<1>(this->children());
  }
  [[nodiscard]] TSolOut& sol_out() {
    return thes::star::get_at<2>(this->children());
  }
  [[nodiscard]] decltype(auto) row_indices() const {
    return thread_info_.indices();
  }

  [[no_unique_address]] DistInfoStorage dist_info_;
  TThreadInfo thread_info_;
  Real relax_;
  Real m1_relax_{Real{1} - relax_};
};

template<typename TDistInfo, typename TReal, AnyMatrix TLhs, AnyVector TSolIn, AnyVector TSolOut,
         AnyVector TRhs>
struct JacobiSweepStorage {
  using Lhs = std::decay_t<TLhs>;
  using Rhs = std::decay_t<TRhs>;
  using Size = SizeIntersection<Lhs, Rhs>;
  using DistInfoStorage = thes::VoidStorage<TDistInfo>;
  static constexpr bool is_shared = std::is_void_v<TDistInfo>;
  static constexpr auto exec_constraints =
    thes::star::joined(merged_exec_constraints<thes::Tuple<TLhs, TSolIn, TSolOut, Rhs>>,
                       thes::Tuple{ThreadKeepNbOrderConstraint<nnz_pattern<Lhs>>{}}) |
    thes::star::to_tuple;

  JacobiSweepStorage(DistInfoStorage&& p_dist_info, TReal p_relax, TLhs&& p_lhs, TSolIn&& p_sol_in,
                     TSolOut&& p_sol_out, TRhs&& p_rhs)
      : dist_info(std::forward<DistInfoStorage>(p_dist_info)), relax(p_relax),
        lhs(std::forward<TLhs>(p_lhs)), sol_in(std::forward<TSolIn>(p_sol_in)),
        sol_out(std::forward<TSolOut>(p_sol_out)), rhs(std::forward<TRhs>(p_rhs)) {}

  auto children() const {
    return std::tie(lhs, sol_in, sol_out, rhs);
  }

  [[no_unique_address]] DistInfoStorage dist_info;
  TReal relax;
  TLhs lhs;
  TSolIn sol_in;
  TSolOut sol_out;
  TRhs rhs;
};

template<typename TDistInfo, typename TReal, typename TLhs, typename TSolIn, typename TSolOut,
         typename TRhs>
struct JacobiSweep;

template<typename TDistInfo, typename TReal, SharedMatrix TLhs, SharedVector TSolIn,
         SharedVector TSolOut, SharedVector TRhs>
struct JacobiSweep<TDistInfo, TReal, TLhs, TSolIn, TSolOut, TRhs>
    : JacobiSweepStorage<TDistInfo, TReal, TLhs, TSolIn, TSolOut, TRhs> {
  using Parent = JacobiSweepStorage<TDistInfo, TReal, TLhs, TSolIn, TSolOut, TRhs>;
  using Size = Parent::Size;
  static constexpr bool is_shared = std::is_void_v<TDistInfo>;

  using Parent::Parent;

  template<typename TThreadInfo>
  [[nodiscard]] auto thread_instance(TThreadInfo&& thread_info, grex::AnyTag auto /*tag*/) {
    using Sink = JacobiSweepSink<thes::VoidConstLvalRef<TDistInfo>, TReal, TThreadInfo, const TLhs&,
                                 const TSolIn&, TSolOut&, const TRhs&>;
    return Sink{
      thes::void_storage_cref(this->dist_info),
      std::forward<TThreadInfo>(thread_info),
      this->relax,
      this->lhs,
      this->sol_in,
      this->sol_out,
      this->rhs,
    };
  }

  [[nodiscard]] Size size() const {
    const auto lhs_size = this->lhs.row_num();
    assert(lhs_size == this->rhs.size());
    return *thes::safe_cast<Size>(lhs_size);
  }

  auto axis_range(thes::AnyIndexTag auto idx) const
  requires(requires { impl::axis_range(this->children(), idx); })
  {
    return impl::axis_range(this->children(), idx);
  }
  decltype(auto) geometry() const
  requires(requires { impl::geometry(this->children()); })
  {
    return impl::geometry(this->children());
  }
};
} // namespace lineal::detail

#endif // INCLUDE_LINEAL_LINEAR_SOLVER_ITERATIVE_STATIONARY_JACOBI_SWEEP_HPP
