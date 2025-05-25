// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_COMPONENT_WISE_SINK_EFFECTIVE_DIFFUSION_HPP
#define INCLUDE_LINEAL_COMPONENT_WISE_SINK_EFFECTIVE_DIFFUSION_HPP

#include <atomic>
#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>

#include "thesauros/macropolis.hpp"
#include "thesauros/ranges.hpp"
#include "thesauros/types.hpp"
#include "thesauros/utility.hpp"

#include "lineal/base.hpp"
#include "lineal/component-wise/facade.hpp"
#include "lineal/parallel.hpp"
#include "lineal/vectorization.hpp"

namespace lineal {
namespace detail {
template<typename TSize, typename TCompressMap>
struct AreaPartSinkConf {
  using Work = void;
  using Value = void;
  using Size = TSize;
  using IndexTag = LocalIndexTag;
  static constexpr bool supports_const_access = false;
  static constexpr bool supports_mutable_access = true;
};

template<typename TReal, typename TSol, typename TCompressMap>
struct EffDiffSinkConf {
  using Work = TReal;
  using Value = void;
  using Size = TSol::Size;
  using IndexTag = LocalIndexTag;
  static constexpr bool supports_const_access = false;
  static constexpr bool supports_mutable_access = true;
};
} // namespace detail

template<typename TValuator, typename TCompressMap>
struct AreaPartSink {
  using Valuator = std::decay_t<TValuator>;
  using Real = Valuator::Real;
  using Size = Valuator::Size;
  using Range = thes::IotaRange<Size>;
  using CompressMap = std::decay_t<TCompressMap>;
  using CompressMapStorage = thes::VoidStorage<TCompressMap>;
  using CompressMapCrefStorage = thes::VoidStorageConstLvalRef<TCompressMap>;
  static constexpr auto is_shared = Valuator::is_shared;

  template<grex::AnyTag TVecTag>
  struct ThreadInstance
      : public facades::SharedNullaryCwOp<ThreadInstance<TVecTag>,
                                          detail::AreaPartSinkConf<Size, CompressMap>> {
    using Counter = grex::TagType<Size, TVecTag>;
    using Parent =
      facades::SharedNullaryCwOp<ThreadInstance, detail::AreaPartSinkConf<Size, CompressMap>>;

    explicit ThreadInstance(const TValuator& valuator, CompressMapCrefStorage compress_map,
                            Range idxs, Size& sum)
        : Parent(idxs.size()), valuator_(valuator), compress_map_(compress_map), idxs_(idxs),
          nb_offset_(valuator.info().after_size(thes::index_tag<0>)), sum_(sum) {}
    ThreadInstance(const ThreadInstance&) = delete;
    ThreadInstance& operator=(const ThreadInstance&) = delete;
    ThreadInstance(ThreadInstance&&) noexcept = delete;
    ThreadInstance& operator=(ThreadInstance&&) noexcept = delete;
    ~ThreadInstance() {
      std::atomic_ref{sum_} += grex::horizontal_add(thread_sum_, TVecTag{});
    }

    template<grex::AnyTag TTag>
    requires(std::is_void_v<TCompressMap> || grex::ScalarTag<TTag>)
    THES_ALWAYS_INLINE constexpr void compute_impl(TTag tag, Size idx) {
      const Size idx1 = idx + idxs_.begin_value();
      const Size idx2 = idx1 + nb_offset_;
      thread_sum_ += valuator_.is_filled_num(idx1, tag) + valuator_.is_filled_num(idx2, tag);
    }

  private:
    const TValuator& valuator_;
    CompressMapCrefStorage compress_map_;
    Range idxs_;
    Size nb_offset_;
    Counter thread_sum_{0};
    Size& sum_;
  };

  explicit AreaPartSink(const TValuator& valuator,
                        thes::VoidStorageRvalRef<TCompressMap> compress_map, Range idxs)
      : valuator_(valuator), compress_map_(std::forward<CompressMapStorage>(compress_map)),
        idxs_(idxs) {}

  template<grex::AnyTag TVecTag>
  ThreadInstance<TVecTag> thread_instance(const auto& /*info*/, TVecTag /*tag*/) {
    return ThreadInstance<TVecTag>{valuator_, compress_map_, idxs_, sum_};
  }

  [[nodiscard]] Size size() const {
    return *thes::safe_cast<Size>(idxs_.size());
  }

  [[nodiscard]] Real area_part() const {
    std::atomic_thread_fence(std::memory_order::seq_cst);
    return Real(sum_) / Real(valuator_.info().after_size(thes::index_tag<0>)) * Real(0.5);
  }

private:
  const TValuator& valuator_;
  CompressMapStorage compress_map_;
  Range idxs_;
  Size sum_{0};
};
template<typename TValuator, typename TCompressMap>
AreaPartSink(const TValuator&, TCompressMap&&,
             thes::IotaRange<typename std::decay_t<TValuator>::Size>)
  -> AreaPartSink<TValuator, thes::UnVoidStorage<TCompressMap>>;

template<typename TReal, typename TValuator, typename TCompressMap, SharedVector TSol>
struct EffDiffSink {
  using Sol = std::decay_t<TSol>;
  using ExtIndex = Sol::ExtIndex;
  using Size = Sol::Size;
  using Range = thes::IotaRange<Size>;
  using CompressMap = std::decay_t<TCompressMap>;
  using CompressMapStorage = thes::VoidStorage<TCompressMap>;
  using CompressMapCrefStorage = thes::VoidStorageConstLvalRef<TCompressMap>;
  static constexpr auto is_shared = Sol::is_shared;
  static constexpr auto exec_constraints = lineal::exec_constraints<Sol>;

  template<grex::AnyTag TVecTag>
  struct ThreadInstance
      : public facades::SharedNullaryCwOp<ThreadInstance<TVecTag>,
                                          detail::EffDiffSinkConf<TReal, Sol, CompressMap>> {
    using Numeric = grex::TagType<TReal, TVecTag>;
    using Parent =
      facades::SharedNullaryCwOp<ThreadInstance, detail::EffDiffSinkConf<TReal, Sol, CompressMap>>;

    explicit ThreadInstance(const TValuator& valuator, CompressMapCrefStorage compress_map,
                            const TSol& sol, Range idxs, TReal& sum)
        : Parent(idxs.size()), valuator_(valuator), compress_map_(compress_map), sol_(sol),
          idxs_(idxs), nb_offset_(valuator.info().after_size(thes::index_tag<0>)), sum_(sum) {}
    ThreadInstance(const ThreadInstance&) = delete;
    ThreadInstance& operator=(const ThreadInstance&) = delete;
    ThreadInstance(ThreadInstance&&) noexcept = delete;
    ThreadInstance& operator=(ThreadInstance&&) noexcept = delete;
    ~ThreadInstance() {
      std::atomic_ref{sum_} += grex::horizontal_add(thread_sum_, TVecTag{});
    }

    template<grex::AnyTag TTag>
    requires(std::is_void_v<TCompressMap> || grex::ScalarTag<TTag>)
    THES_ALWAYS_INLINE constexpr void compute_impl(TTag tag, Size index) {
      if constexpr (std::is_void_v<TCompressMap>) {
        const Size index1 = index + idxs_.begin_value();
        const auto sol1 = grex::load_ptr(sol_.data() + index1, tag);
        const Size index2 = index1 + nb_offset_;
        const auto sol2 = grex::load_ptr(sol_.data() + index2, tag);

        thread_sum_ += valuator_.diffusion_coeff(index1, index2, tag) * (sol2 - sol1);
      } else {
        static_assert(grex::ScalarTag<decltype(tag)>);

        const Size index1 = index + idxs_.begin_value();
        const auto key1 = compress_map_[index1];
        const Size index2 = index1 + nb_offset_;
        const auto key2 = compress_map_[index2];

        if (key1.has_value() && key2.has_value()) {
          // TODO Somewhat unsafe: Is this the same type used when storing the indices?
          const auto sol2 = sol_[ExtIndex{key2.value()}];
          const auto sol1 = sol_[ExtIndex{key1.value()}];
          thread_sum_ += valuator_.diffusion_coeff(index1, index2, tag) * (sol2 - sol1);
        }
      }
    }

  private:
    const TValuator& valuator_;
    CompressMapCrefStorage compress_map_;
    const TSol& sol_;
    Range idxs_;
    Size nb_offset_;
    Numeric thread_sum_{0};
    TReal& sum_;
  };

  explicit EffDiffSink(const TValuator& valuator,
                       thes::VoidStorageRvalRef<TCompressMap> compress_map, const TSol& sol,
                       Range idxs, thes::TypeTag<TReal> /*tag*/)
      : valuator_(valuator), compress_map_(std::forward<CompressMapStorage>(compress_map)),
        sol_(sol), idxs_(idxs) {}

  template<grex::AnyTag TVecTag>
  ThreadInstance<TVecTag> thread_instance(const auto& /*info*/, TVecTag /*tag*/) {
    return ThreadInstance<TVecTag>{valuator_, compress_map_, sol_, idxs_, sum_};
  }

  [[nodiscard]] Size size() const {
    return *thes::safe_cast<Size>(idxs_.size());
  }

  [[nodiscard]] TReal sum() const {
    std::atomic_thread_fence(std::memory_order::seq_cst);
    return sum_;
  }

private:
  const TValuator& valuator_;
  CompressMapStorage compress_map_;
  const TSol& sol_;
  Range idxs_;
  TReal sum_{0};
};
template<typename TReal, typename TValuator, typename TCompressMap, SharedVector TSol>
EffDiffSink(const TValuator&, TCompressMap&&, const TSol&,
            thes::IotaRange<typename std::decay_t<TSol>::Size>, thes::TypeTag<TReal>)
  -> EffDiffSink<TReal, TValuator, thes::UnVoidStorage<TCompressMap>, TSol>;

template<typename TReal>
struct EffDiffCalc {
  template<typename TValuator>
  requires(TValuator::is_shared)
  static auto area_part(const TValuator& valuator, const auto& map, const Env auto& env) {
    using Size = TValuator::Size;
    static constexpr std::size_t dimension_num = TValuator::dimension_num;
    static constexpr std::size_t flow_axis = TValuator::flow_axis;
    static_assert(dimension_num > 1 && flow_axis == 0);

    decltype(auto) info = valuator.info();

    const Size mid = info.axis_size(thes::index_tag<0>) / 2;
    const Size other_num = info.after_size(thes::index_tag<0>);
    auto range = thes::range(*thes::safe_cast<Size>(mid * other_num),
                             *thes::safe_cast<Size>((mid + 1) * other_num));

    AreaPartSink sink(valuator, map, range);
    env.execution_policy().execute(sink);
    const auto area_part = sink.area_part();
    env.log("area_part", area_part);
    return area_part;
  }
  template<typename TValuator>
  requires(!TValuator::is_shared)
  static auto area_part(const TValuator& valuator, const auto& map, const Env auto& env) {
    using Real = TValuator::Real;
    using Size = TValuator::Size;
    using GlobalSize = TValuator::DistributedInfo::GlobalSize;

    const auto& dist_info = valuator.distributed_info();

    const Real local_area_part = [&] {
      static constexpr std::size_t dimension_num = TValuator::dimension_num;
      static constexpr std::size_t flow_axis = TValuator::flow_axis;
      static_assert(dimension_num > 1 && flow_axis == 0);

      decltype(auto) info = valuator.info();

      const Size mid = info.axis_size(thes::index_tag<0>) / 2;
      const Size other_num = info.after_size(thes::index_tag<0>);

      const auto global_work_range =
        thes::range(*thes::safe_cast<GlobalSize>(mid * other_num),
                    *thes::safe_cast<GlobalSize>((mid + 1) * other_num)) &
        dist_info.index_range_within(own_index_tag, global_index_tag);
      const auto local_work_range =
        dist_info.convert(global_work_range, global_index_tag, local_index_tag);

      if (local_work_range.is_empty()) {
        return Real{0};
      }

      AreaPartSink sink(valuator, map, local_work_range);
      env.execution_policy().execute(sink);
      return sink.area_part();
    }();

    const auto global_area_part = dist_info.communicator().allreduce(local_area_part, std::plus{});
    env.log("global_area_part", global_area_part);
    return global_area_part;
  }

  template<typename TCompressMap>
  EffDiffCalc(const auto& valuator, TCompressMap&& map, const Env auto& env)
      : area_part_(
          area_part(valuator, std::forward<TCompressMap>(map), env.add_object("eff_diff_calc"))) {}
  EffDiffCalc(const auto& valuator, const Env auto& env)
      : EffDiffCalc(valuator, thes::Empty{}, env) {}

  template<typename TValuator, typename TCompressMap>
  auto operator()(const TValuator& valuator, TCompressMap&& map, const SharedVector auto& sol,
                  const Env auto& env) const {
    using Real = TValuator::Real;
    using Size = TValuator::Size;
    static constexpr std::size_t dimension_num = TValuator::dimension_num;
    static constexpr std::size_t flow_axis = TValuator::flow_axis;
    static_assert(dimension_num > 1 && flow_axis == 0);

    decltype(auto) info = valuator.info();

    const auto eff_diff = [&] {
      const Size mid = info.axis_size(thes::index_tag<0>) / 2;
      const Size other_num = info.after_size(thes::index_tag<0>);
      auto range = thes::range(*thes::safe_cast<Size>(mid * other_num),
                               *thes::safe_cast<Size>((mid + 1) * other_num));

      EffDiffSink sink{
        valuator, std::forward<TCompressMap>(map), sol, range, thes::type_tag<Real>,
      };
      env.execution_policy().execute(sink);
      return sink.sum();
    }();

    Real out = eff_diff;
    out *= info.axis_quotient(thes::index_tag<0>);
    out /= area_part_ * info.other_length(thes::index_tag<0>);
    out *= info.axis_length(thes::index_tag<0>);
    return out / (info.solution_end() - info.solution_start());
  }

  auto operator()(const auto& valuator, const AnyVector auto& solution,
                  const auto& executor) const {
    return (*this)(valuator, thes::Empty{}, solution, executor);
  }

private:
  TReal area_part_;
};

template<typename TVal, typename TCM>
EffDiffCalc(const TVal&, TCM&&, const Env auto&) -> EffDiffCalc<typename TVal::Real>;
template<typename TVal, typename TCM>
EffDiffCalc(const TVal&, TCM&&, const auto&, const Env auto&) -> EffDiffCalc<typename TVal::Real>;

template<typename TVal>
EffDiffCalc(const TVal&, const Env auto&) -> EffDiffCalc<typename TVal::Real>;
} // namespace lineal

#endif // INCLUDE_LINEAL_COMPONENT_WISE_SINK_EFFECTIVE_DIFFUSION_HPP
