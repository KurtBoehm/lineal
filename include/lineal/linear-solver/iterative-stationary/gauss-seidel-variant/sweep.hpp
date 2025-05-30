// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_LINEAR_SOLVER_ITERATIVE_STATIONARY_GAUSS_SEIDEL_VARIANT_SWEEP_HPP
#define INCLUDE_LINEAL_LINEAR_SOLVER_ITERATIVE_STATIONARY_GAUSS_SEIDEL_VARIANT_SWEEP_HPP

#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

#include "thesauros/algorithms.hpp"
#include "thesauros/macropolis.hpp"
#include "thesauros/ranges.hpp"
#include "thesauros/static-ranges.hpp"
#include "thesauros/types.hpp"
#include "thesauros/utility.hpp"

#include "lineal/base.hpp"
#include "lineal/component-wise/exec-constraint.hpp"
#include "lineal/component-wise/facade.hpp"
#include "lineal/parallel.hpp"
#include "lineal/vectorization.hpp"

namespace lineal {
// regular: The normal SOR method.
// ultra: The variant defined in “Multigrid Smoothers for Ultra-Parallel Computing”, Section 6.1,
//        available at https://www.osti.gov/servlets/purl/1117969,
//        and implemented in hypre in parcsr_ls/par_relax.h.
enum struct SorVariant : thes::u8 { regular, ultra };

namespace detail {
namespace sor {
template<typename TRange>
struct RangeSelector;

template<typename T>
struct RangeSelector<thes::IotaRange<T>> {
  using Range = thes::IotaRange<T>;

  THES_ALWAYS_INLINE static auto mask(const Range& range, T start, grex::VectorTag auto tag) {
    constexpr auto idx_offs = thes::star::typed_iota<T, 0, tag.size> |
                              thes::star::transform([](T idx) { return idx; }) |
                              thes::star::to_array;
    auto idx = start + grex::to_vector(idx_offs);
    return (range.begin_value() <= idx) & (idx < range.end_value());
  }

  THES_ALWAYS_INLINE static auto select(const Range& range, T start, auto true_op, auto false_op,
                                        grex::VectorTag auto tag) {
    using Vec = thes::TypeSeq<decltype(true_op(tag)), decltype(false_op(tag))>::Unique;
    using Value = Vec::Value;

    auto end = start + tag.part();
    if (range.begin_value() <= start && end <= range.end_value()) [[likely]] {
      return true_op(tag);
    }
    if (end <= range.begin_value() || range.end_value() <= start) {
      return false_op(tag);
    }

    const auto m = grex::convert_safe<Value>(mask(range, start, tag), tag);
    return grex::select(m, true_op(tag), false_op(tag), tag);
  }
};

template<typename TRange>
THES_ALWAYS_INLINE inline auto select(const TRange& range, auto idx, auto true_op, auto false_op,
                                      grex::VectorTag auto tag) {
  return RangeSelector<TRange>::select(range, std::move(idx), std::move(true_op),
                                       std::move(false_op), tag);
}
template<typename TRange>
THES_ALWAYS_INLINE inline auto mask(const TRange& range, auto idx, grex::VectorTag auto tag) {
  return RangeSelector<TRange>::mask(range, std::move(idx), tag);
}
} // namespace sor

struct SorSweepConf {
  using Work = void;
  using Value = void;
  static constexpr bool supports_const_access = false;
  static constexpr bool supports_mutable_access = true;
  static constexpr auto component_wise_seq = thes::Tuple{thes::index_tag<0>, thes::index_tag<3>};
};

template<typename TDistInfo, SorVariant tVariant, AnyDirectionTag auto tDir, typename TReal,
         typename TThreadInfo, SharedMatrix TLhs, SharedVector TSolIn, SharedVector TSolOut,
         SharedVector TRhs>
struct SorSweepSink
    : public facades::ComponentWiseOp<
        SorSweepSink<TDistInfo, tVariant, tDir, TReal, TThreadInfo, TLhs, TSolIn, TSolOut, TRhs>,
        SorSweepConf, TLhs, TSolIn, TSolOut, TRhs> {
  using Real = TReal;
  using Value = Real;
  using Lhs = std::decay_t<TLhs>;
  using Rhs = std::decay_t<TRhs>;
  using LhsValue = Lhs::Value;
  using InValue = std::decay_t<TSolIn>::Value;
  using OutValue = std::decay_t<TSolOut>::Value;
  static constexpr bool is_shared = std::is_void_v<TDistInfo>;
  static constexpr thes::IterDirection iter_dir =
    thes::StaticMap{
      thes::static_kv<forward_tag, thes::IterDirection::FORWARD>,
      thes::static_kv<backward_tag, thes::IterDirection::BACKWARD>,
    }
      .get(thes::auto_tag<tDir>);

  using Parent = facades::ComponentWiseOp<SorSweepSink, SorSweepConf, TLhs, TSolIn, TSolOut, TRhs>;
  using DistInfoStorage = thes::VoidStorage<TDistInfo>;

  using Size = SizeIntersection<Lhs, Rhs>;
  using LocalIdx = OptLocalIndex<is_shared, Size>;

  explicit SorSweepSink(DistInfoStorage&& dist_info, TThreadInfo&& thread_info, Real relax,
                        TLhs&& lhs, TSolIn&& sol_in, TSolOut&& sol_out, TRhs&& rhs)
      : Parent(std::forward<TLhs>(lhs), std::forward<TSolIn>(sol_in),
               std::forward<TSolOut>(sol_out), std::forward<TRhs>(rhs)),
        dist_info_(std::forward<DistInfoStorage>(dist_info)),
        thread_info_(std::forward<TThreadInfo>(thread_info)), relax_(relax) {}

  THES_ALWAYS_INLINE void compute_base_scalar(auto lhs_row, auto rhs_val) {
    decltype(auto) row_idxs = row_indices();
    decltype(auto) col_idxs = [&]() -> decltype(auto) {
      if constexpr (is_shared) {
        return row_idxs;
      } else {
        return dist_info_.convert(row_idxs, own_index_tag, local_index_tag);
      }
    }();

    const TSolIn& x_in = sol_in();
    TSolOut& x_out = sol_out();
    Real accum = rhs_val;
    Real diag = std::numeric_limits<Real>::signaling_NaN();
    Real div{};

    auto update = [&](LocalIdx j, Real aij, const auto& vec) THES_ALWAYS_INLINE {
      accum = grex::fnma(aij, Real(vec[j]), accum, grex::Scalar{});
    };
    auto update_conditional = [&](thes::AnyBoolTag auto is_conditional, LocalIdx j, Real aij)
                                THES_ALWAYS_INLINE {
                                  bool contains{};
                                  if constexpr (tVariant == SorVariant::ultra || is_conditional) {
                                    contains = col_idxs.contains(index_value(j));
                                  }
                                  if constexpr (tVariant == SorVariant::ultra) {
                                    div += contains ? Real{} : std::abs(aij);
                                  }
                                  if constexpr (is_conditional) {
                                    if (contains) [[likely]] {
                                      return update(j, aij, x_out);
                                    }
                                  }
                                  update(j, aij, x_in);
                                };

    lhs_row.iterate(
      // j < index
      [&](LocalIdx j, Real aij)
        THES_ALWAYS_INLINE { update_conditional(thes::bool_tag<(tDir == forward_tag)>, j, aij); },
      // j == index
      [&]([[maybe_unused]] LocalIdx j, Real aij) THES_ALWAYS_INLINE {
        diag = aij;
        if constexpr (tVariant == SorVariant::ultra) {
          update(j, aij, x_in);
        }
        assert(col_idxs.contains(index_value(j)));
      },
      // j > index
      [&](LocalIdx j, Real aij)
        THES_ALWAYS_INLINE { update_conditional(thes::bool_tag<(tDir == backward_tag)>, j, aij); },
      valued_tag, unordered_tag);

    const LocalIdx i = lhs_row.ext_index();
    assert(std::isfinite(diag));
    if constexpr (tVariant == SorVariant::ultra) {
      x_out[i] = OutValue(Real(x_in[i]) + relax_ * accum / (std::abs(diag) + 0.5 * div));
    } else {
      const auto v = grex::fma(relax_, accum / diag, m1_relax_ * Real(x_in[i]), grex::Scalar{});
      x_out[i] = OutValue(v);
    }
  }

  template<grex::FullVectorTag TVecTag>
  THES_ALWAYS_INLINE void compute_base_multicolumn(auto row, auto rhs_val, TVecTag ftag) {
    using InMaskTag = grex::TypedVectorMaskSize<InValue, ftag.size>;
    using OutMaskTag = grex::TypedVectorMaskSize<OutValue, ftag.size>;
    constexpr bool is_forward = tDir == forward_tag;

    decltype(auto) row_idxs = row_indices();
    decltype(auto) col_idxs = [&]() -> decltype(auto) {
      if constexpr (is_shared) {
        return row_idxs;
      } else {
        return dist_info_.convert(row_idxs, own_index_tag, local_index_tag);
      }
    }();

    const LocalIdx i = row.ext_index();
    const auto ivec = grex::constant(index_value(i), ftag);

    const TSolIn& x_in = sol_in();
    TSolOut& x_out = sol_out();
    const Real x_diag(x_in[i]);

    const Real rrhs(rhs_val);
    auto accum_vec = grex::constant(rrhs / ftag.size, ftag);
    auto dia_vec = grex::constant(Real{0}, ftag);
    // TODO GENERALIZE TO OTHER COLUMN INDICES!
    const auto limit =
      grex::constant(is_forward ? col_idxs.begin_value() : col_idxs.end_value(), ftag);
    row.multicolumn_iterate(
      [&](auto j, auto aij, auto vtag) THES_ALWAYS_INLINE {
        const auto j_les_lim = j < limit;
        const auto j_geq_lim = j >= limit;
        const auto j_les_i = j < ivec;
        const auto j_gre_i = j > ivec;

        // TODO Formalize the assumption that masked-out values are 0?!

        const auto new_mask = vtag.mask(is_forward ? (j_geq_lim & j_les_i) : (j_gre_i & j_les_lim));
        const auto new_vec =
          grex::convert_unsafe<Real>(x_out.lookup(j, OutMaskTag{new_mask}), vtag);

        const auto old_mask = vtag.mask(is_forward ? (((j != ivec) & j_les_lim) | j_gre_i)
                                                   : (j_les_i | ((j != ivec) & j_geq_lim)));
        const auto old_vec = grex::convert_unsafe<Real>(x_in.lookup(j, InMaskTag{old_mask}), vtag);

        const auto aij_real = grex::convert_unsafe<Real>(aij, vtag);
        accum_vec = grex::fnma(aij_real, new_vec + old_vec, accum_vec, ftag);
        dia_vec = grex::select(vtag.mask(j == ivec), aij_real, dia_vec, ftag);
      },
      valued_tag, unordered_tag, ftag);

    const Real accum = grex::horizontal_add(accum_vec, ftag);
    const Real diag = grex::horizontal_add(dia_vec, ftag);
    assert(diag != 0);
    const Real quot = accum / diag;

    x_out[i] = OutValue(grex::fma(relax_, quot, m1_relax_ * x_diag, grex::Scalar{}));
  }

  template<grex::VectorTag TVecTag>
  THES_ALWAYS_INLINE void compute_base_simd(auto row, auto rhs_vec, TVecTag tag) {
    using Vec = grex::Vector<Real, tag.size>;
    static constexpr std::size_t dimension_num = Lhs::dimension_num;

    decltype(auto) row_idxs = row_indices();
    decltype(auto) col_idxs = [&]() -> decltype(auto) {
      if constexpr (is_shared) {
        return row_idxs;
      } else {
        return dist_info_.convert(row_idxs, global_index_tag, local_index_tag);
      }
    }();

    const TSolIn& x_in = sol_in();
    auto it_in = x_in.iter_at(row.index());
    TSolOut& x_out = sol_out();
    auto it_out = x_out.iter_at(row.index());

    const LocalIdx i = row.ext_index();
    const Size ival = index_value(i);
    auto accum = grex::convert_unsafe<Real>(rhs_vec, tag);

#ifdef FINITE_CHECK
#error "FINITE_CHECK is already defined!"
#endif
#define FINITE_CHECK(x) assert(grex::horizontal_and(grex::is_finite(x, tag), tag))

    Vec diag{};
    Vec nb{};
    Vec div{};
    if constexpr (tVariant == SorVariant::ultra) {
      div = grex::zero<Real>(tag);
    }

    auto update = [&](auto aij, auto vec_vec) THES_ALWAYS_INLINE {
      FINITE_CHECK(aij);
      FINITE_CHECK(vec_vec);
      accum = grex::fnma(aij, grex::convert_unsafe<Real>(vec_vec, tag), accum, tag);
      FINITE_CHECK(accum);
    };
    auto update_conditional = [&](thes::AnyBoolTag auto is_conditional, auto aij, auto off,
                                  thes::AnyIndexTag auto dim) {
      if constexpr (is_conditional) {
        if constexpr (dim + 1 == dimension_num) {
          nb = aij;
        } else {
          auto off_vec = sor::select(
            col_idxs, off(ival),
            [&](auto t) THES_ALWAYS_INLINE {
              return grex::convert_unsafe<Real>(off(it_out).compute(t), tag);
            },
            [&](auto t)
              THES_ALWAYS_INLINE { return grex::convert_unsafe<Real>(off(it_in).compute(t), tag); },
            tag);
          update(aij, off_vec);
        }
      } else {
        update(aij, off(it_in).compute(tag));
      }
      if constexpr (tVariant == SorVariant::ultra) {
        div += grex::select(grex::convert_safe<Real>(sor::mask(col_idxs, off(ival), tag), tag),
                            grex::zero<Real>(tag), grex::abs(aij, tag), tag);
      }
    };

    // TODO The dimension parameter is inconsistent with the interface described in the paper!
    row.banded_iterate(
      // j < index
      [&](auto aij, auto off, thes::AnyIndexTag auto dim) THES_ALWAYS_INLINE {
        update_conditional(thes::bool_tag<(tDir == forward_tag)>, aij, off, dim);
      },
      // j == index
      [&](auto aij, auto /*off*/) THES_ALWAYS_INLINE {
        diag = aij;
        if constexpr (tVariant == SorVariant::ultra) {
          update(aij, it_in.compute(tag));
        }
        // TODO assert(col_idxs.contains(index_value(j)));
      },
      // j > index
      [&](auto aij, auto off, thes::AnyIndexTag auto dim) THES_ALWAYS_INLINE {
        update_conditional(thes::bool_tag<(tDir == backward_tag)>, aij, off, dim);
      },
      unordered_tag, tag);

    if constexpr (tVariant == SorVariant::ultra) {
      div = grex::abs(diag, tag) + Real(0.5) * div;
    } else {
      div = diag;
    }

    const auto vin = grex::convert_unsafe<Real>(x_in.compute(i, tag), tag);
    FINITE_CHECK(vin);
    Vec add{};
    const auto quot = relax_ / div;
    if constexpr (tVariant == SorVariant::ultra) {
      add = vin + accum * quot;
    } else {
      add = grex::fma(accum, quot, m1_relax_ * vin, tag);
    }
    FINITE_CHECK(add);
    auto factor = nb * quot;
    FINITE_CHECK(factor);

    Vec x{};
    if constexpr (tDir == forward_tag) {
      Real first = thes::make_finite(*(it_out - 1));
      assert(std::isfinite(first));
      x = grex::fnma(factor, Vec::expand_zero(first), add, tag);
      FINITE_CHECK(x);
      for (std::size_t k = 1; k < tag.size; ++k) {
        x = grex::fnma(factor, grex::shuffle_up(x, first, tag), add, tag);
        FINITE_CHECK(x);
      }
    } else {
      Real last;
      if constexpr (grex::PartialVectorTag<TVecTag>) {
        auto tagin = tag.instantiate(thes::type_tag<Real>);
        add = tagin.mask(add);
        factor = grex::select(tagin.mask(), factor, Vec{1}, tag);
        last = 0;
        x = Vec{0};
      } else {
        last = thes::make_finite(*(it_out + tag.part()));
        x = Vec{0};
        x.insert(tag.part() - 1, last);
      }
      FINITE_CHECK(x);

      x = grex::fnma(factor, x, add, tag);
      FINITE_CHECK(x);
      for (std::size_t k = 1; k < tag.size; ++k) {
        x = grex::fnma(factor, grex::shuffle_down(x, last, tag), add, tag);
        FINITE_CHECK(x);
      }
    }

    FINITE_CHECK(x);
    x_out.store(i, grex::convert_unsafe<OutValue>(x, tag), tag);

#undef FINITE_CHECK
  }

  template<grex::AnyTag TTag>
  THES_ALWAYS_INLINE void compute_base(auto lhs_off, auto rhs_off, auto rhs_load, TTag tag) {
    if constexpr (grex::VectorTag<TTag> && grex::is_geometry_respecting<TTag> &&
                  BandedIterable<Lhs, TTag>) {
      compute_base_simd(lhs_off(0), rhs_load(), tag);
    } else {
      grex::for_each<Size>([&](auto i)
                             THES_ALWAYS_INLINE { compute_base_scalar(lhs_off(i), rhs_off(i)); },
                           thes::auto_tag<iter_dir>, tag);
    }
  }

  THES_ALWAYS_INLINE void compute_impl(grex::AnyTag auto tag, const auto& /*children*/, auto lhs_it,
                                       auto rhs_it) {
    compute_base([&](auto i) THES_ALWAYS_INLINE { return lhs_it[i]; },
                 [&](auto i) THES_ALWAYS_INLINE { return rhs_it[i]; },
                 [&]() THES_ALWAYS_INLINE { return rhs_it.compute(tag); }, tag);
  }
  THES_ALWAYS_INLINE void compute_impl(grex::AnyTag auto tag, const auto& arg,
                                       const auto& /*children*/, const auto& lhs, const auto& rhs) {
    compute_base([&](Size i) THES_ALWAYS_INLINE { return lhs[arg + i]; },
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

template<typename TDistInfo, AnyDirectionTag auto tDir, typename TReal, AnyMatrix TLhs,
         AnyVector TSolIn, AnyVector TSolOut, AnyVector TRhs>
struct SorSweepStorage {
  using Lhs = std::decay_t<TLhs>;
  using Rhs = std::decay_t<TRhs>;
  using Size = SizeIntersection<Lhs, Rhs>;
  using DistInfoStorage = thes::VoidStorage<TDistInfo>;
  static constexpr bool is_shared = std::is_void_v<TDistInfo>;
  static constexpr auto exec_constraints =
    thes::star::joined(merged_exec_constraints<thes::Tuple<TLhs, TSolIn, TSolOut, Rhs>>,
                       thes::Tuple{ThreadKeepNbOrderConstraint<nnz_pattern<Lhs>>{},
                                   IterationDirectionConstraint<tDir>{}}) |
    thes::star::to_tuple;

  SorSweepStorage(DistInfoStorage&& p_dist_info, TReal p_relax, TLhs&& p_lhs, TSolIn&& p_sol_in,
                  TSolOut&& p_sol_out, TRhs&& p_rhs, std::optional<std::size_t> min_per_thread)
      : dist_info(std::forward<DistInfoStorage>(p_dist_info)), relax(p_relax),
        lhs(std::forward<TLhs>(p_lhs)), sol_in(std::forward<TSolIn>(p_sol_in)),
        sol_out(std::forward<TSolOut>(p_sol_out)), rhs(std::forward<TRhs>(p_rhs)),
        min_per_thread(min_per_thread) {}

  auto children() const {
    return std::tie(lhs, sol_in, sol_out, rhs);
  }

  [[no_unique_address]] DistInfoStorage dist_info;
  TReal relax;
  TLhs lhs;
  TSolIn sol_in;
  TSolOut sol_out;
  TRhs rhs;
  std::optional<std::size_t> min_per_thread;
};

template<typename TDistInfo, SorVariant tVariant, AnyDirectionTag auto tDir, typename TReal,
         typename TLhs, typename TSolIn, typename TSolOut, typename TRhs>
struct SorSweep;

template<typename TDistInfo, SorVariant tVariant, AnyDirectionTag auto tDir, typename TReal,
         SharedMatrix TLhs, SharedVector TSolIn, SharedVector TSolOut, SharedVector TRhs>
struct SorSweep<TDistInfo, tVariant, tDir, TReal, TLhs, TSolIn, TSolOut, TRhs>
    : SorSweepStorage<TDistInfo, tDir, TReal, TLhs, TSolIn, TSolOut, TRhs> {
  using Parent = SorSweepStorage<TDistInfo, tDir, TReal, TLhs, TSolIn, TSolOut, TRhs>;
  using Size = Parent::Size;
  static constexpr bool is_shared = std::is_void_v<TDistInfo>;

  using Parent::Parent;

  template<typename TThreadInfo>
  [[nodiscard]] auto thread_instance(TThreadInfo&& thread_info, grex::AnyTag auto /*tag*/) {
    using Sink = SorSweepSink<thes::VoidConstLvalRef<TDistInfo>, tVariant, tDir, TReal, TThreadInfo,
                              const TLhs&, const TSolIn&, TSolOut&, const TRhs&>;
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
  [[nodiscard]] thes::Optional<std::size_t> thread_num(std::size_t expo_size) const {
    return thes::Optional{this->min_per_thread}.transform(
      [&](std::size_t m) { return std::clamp<std::size_t>(this->size() / m, 1, expo_size); });
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
} // namespace detail
} // namespace lineal

#endif // INCLUDE_LINEAL_LINEAR_SOLVER_ITERATIVE_STATIONARY_GAUSS_SEIDEL_VARIANT_SWEEP_HPP
