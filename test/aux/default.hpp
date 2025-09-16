// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef TEST_AUX_STENCIL_HPP
#define TEST_AUX_STENCIL_HPP

#include <concepts>
#include <cstddef>
#include <optional>
#include <type_traits>
#include <utility>

#include "thesauros/execution.hpp"
#include "thesauros/io.hpp"
#include "thesauros/memory.hpp"
#include "thesauros/static-ranges.hpp"
#include "thesauros/types.hpp"
#include "thesauros/utility.hpp"

#include "lineal/base.hpp"
#include "lineal/environment.hpp"
#include "lineal/linear-solver.hpp"
#include "lineal/parallel.hpp"
#include "lineal/tensor.hpp"
#include "lineal/vectorization.hpp"

namespace lineal::test {
struct DefaultSharedDefs {
  using Self = DefaultSharedDefs;

  using Real = thes::f64;
  using LoReal = thes::f32;
  using SizeByte = thes::ByteInteger<4>;
  using Size = SizeByte::Unsigned;
  using NonZeroSizeByte = thes::ByteInteger<5>;
  using NonZeroSize = NonZeroSizeByte::Unsigned;

  static constexpr std::size_t flow_axis = 0;
  static constexpr std::size_t dimension_num = 3;
  static constexpr std::size_t max_material_num = 4;

  struct BaseDefs {
    using Real = Self::Real;
    using LoReal = Self::LoReal;
    using SizeByte = Self::SizeByte;
    using Size = Self::Size;
    using NonZeroSizeByte = Self::NonZeroSizeByte;
    using NonZeroSize = Self::NonZeroSize;

    static constexpr std::size_t flow_axis = Self::flow_axis;
    static constexpr std::size_t dimension_num = Self::dimension_num;
    static constexpr std::size_t max_material_num = Self::max_material_num;
  };

  template<typename T>
  using Alloc = thes::HugePagesAllocator<T>;
  using ByteAlloc = Alloc<std::byte>;

  using LookupSymmetricDiffInfo = LookupSymmetricDiffusionInfo<BaseDefs>;
  template<typename TReal>
  using DenseVector =
    lineal::DenseVector<TReal, Size, grex::native_sizes<TReal>.back(), Alloc<TReal>>;

  struct Defs : public BaseDefs {
    using SystemInfo = const LookupSymmetricDiffInfo&;
    using CellInfo = thes::u8;
    using MaterialIndices = const CellStorage<CellInfo, Alloc<CellInfo>>&;
    using LookupWright = SymmetricDiffusionLookupWright;

    using HiVector = DenseVector<Real>;
    using LoVector = DenseVector<LoReal>;
  };

  using LookupValuatorZero = StationaryDiffusionLookupValuator<Defs, ZeroToZeroTag, void>;
  using LookupValuator = StationaryDiffusionLookupValuator<Defs, ZeroToOneTag, void>;
  using ValueValuatorZero = StationaryDiffusionValueValuator<Defs, ZeroToZeroTag, void, ByteAlloc>;
  using ValueValuator = StationaryDiffusionValueValuator<Defs, ZeroToOneTag, void, ByteAlloc>;

  template<typename TValuator>
  using StencilMatrix = AdjacentStencilMatrix<const TValuator&>;
  template<typename TValuator>
  using StencilVector = AdjacentStencilVector<const TValuator&>;
  template<typename TReal>
  using CsrMatrix = lineal::CsrMatrix<TReal, SizeByte, NonZeroSizeByte, Alloc<TReal>>;

  using DuneDepCriterion = amg::DuneDependencyCriterion<Real>;
  using DepDetective = DependencyDetective<DuneDepCriterion>;

  template<bool tIsShared>
  using AggMap = amg::BaseMultithreadAggregateMap<tIsShared, SizeByte, ByteAlloc>;

  template<bool tIsShared>
  using HomoAggregator = amg::HomogeneousAggregator<AggMap<tIsShared>>;
  template<bool tIsShared>
  using RefiAggregator = amg::RefinedAggregator<AggMap<tIsShared>>;
  template<bool tIsShared>
  using FileAggregator = amg::FileAggregator<AggMap<tIsShared>>;

  template<bool tIsShared>
  using HomoMatAggregator = amg::MatrixAggregator<DepDetective, HomoAggregator<tIsShared>,
                                                  amg::NegativeOffdiagonalTransform, ByteAlloc>;
  template<bool tIsShared>
  using RefiMatAggregator = amg::MatrixAggregator<DepDetective, RefiAggregator<tIsShared>,
                                                  amg::NegativeOffdiagonalTransform, ByteAlloc>;
  template<bool tIsShared>
  using FileMatAggregator = amg::MatrixAggregator<DepDetective, FileAggregator<tIsShared>,
                                                  amg::NegativeOffdiagonalTransform, ByteAlloc>;

  template<typename TThreadNum = thes::Empty,
           thes::AnyIndexTag TVecSize = thes::IndexTag<grex::native_sizes<Real>.back()>,
           thes::AnyBoolTag TTiling = thes::TrueTag>
  static auto make_expo(TThreadNum thread_num = {}, TVecSize vec_size = {}, TTiling tiling = {}) {
    auto mkexec = [&] {
      if constexpr (std::same_as<TThreadNum, thes::Empty>) {
        return thes::SequentialExecutor{};
      } else {
        return thes::FixedStdThreadPool(thread_num);
      }
    };

    using Executor = decltype(mkexec());
    using TileSizes =
      std::conditional_t<tiling, thes::star::Constant<3, thes::ValueTag<Size, 8>>, void>;
    using ExPo =
      std::conditional_t<vec_size == 1, AdaptivePolicy<Executor, TileSizes, grex::scalar_tag>,
                         AdaptivePolicy<Executor, TileSizes, grex::full_tag<vec_size>>>;
    return ExPo{std::in_place, mkexec, std::nullopt};
  }

  template<typename TThreadNum = thes::Empty,
           thes::AnyIndexTag TVecSize = thes::IndexTag<grex::native_sizes<Real>.back()>,
           thes::AnyBoolTag TTiling = thes::TrueTag>
  static auto make_env(TThreadNum thread_num = {}, TVecSize vec_size = {}, TTiling tiling = {}) {
    auto mkexpo = [&] { return make_expo(thread_num, vec_size, tiling); };
    return Environment{std::in_place, mkexpo, JsonLogger{thes::Indentation{2}, unflush_tag}};
  }

  template<AnyVector TFineVec, AnyMatrix TCoarseMat, AnyVector TCoarseVec, IsSymmetric tIsSymmetric>
  static auto make_wright() {
    static constexpr bool is_shared = SharedTensors<TCoarseMat, TCoarseVec>;
    static constexpr auto sor_var = lineal::SorVariant::ultra;

    using PreSmoother = lineal::SorSolver<Real, void, lineal::forward_tag, sor_var>;
    using PostSmoother = lineal::SorSolver<Real, void, lineal::backward_tag, sor_var>;
    using CoarseSolver = std::conditional_t<tIsSymmetric == IsSymmetric{true},
                                            lineal::CholeskySolver<Real, TCoarseMat>,
                                            lineal::LuSolver<Real, TCoarseMat>>;

    using HierarchyWright =
      lineal::HierarchyWright</*TWork=*/Real, /*TCoarseLhs=*/TCoarseMat,
                              /*TCoarseRhs=*/TCoarseVec, /*TCoarseSol=*/TCoarseVec,
                              /*TFineAuxVec=*/TFineVec,
                              /*TCoarseAuxVec=*/TCoarseVec, /*TByteAlloc=*/ByteAlloc>;

    const HierarchyWright wright{{
      /*max_level_num=*/16,
      /*min_coarsen_size=*/32,
      /*min_size_per_process=*/48,
      /*min_coarsen_factor=*/1.2,
    }};
    return wright.instantiate(
      HomoMatAggregator<is_shared>{
        DepDetective{{
          /*strong_threshold=*/1.0 / 3.0,
          /*aggressive_strong_threshold=*/1.0 / 3.0,
        }},
        HomoAggregator<is_shared>{{
          .min_aggregate_size = 6,
          .max_aggregate_size = 9,
          .min_per_thread = 8,
        }},
      },
      PreSmoother{/*relax=*/1.0}, PostSmoother{/*relax=*/1.0}, CoarseSolver{});
  }
};

} // namespace lineal::test

#endif // TEST_AUX_STENCIL_HPP
