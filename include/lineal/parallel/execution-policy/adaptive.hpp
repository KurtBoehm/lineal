// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_PARALLEL_EXECUTION_POLICY_ADAPTIVE_HPP
#define INCLUDE_LINEAL_PARALLEL_EXECUTION_POLICY_ADAPTIVE_HPP

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <optional>
#include <type_traits>
#include <utility>

#include "thesauros/algorithms.hpp"
#include "thesauros/macropolis.hpp"
#include "thesauros/math.hpp"
#include "thesauros/ranges.hpp"
#include "thesauros/static-ranges.hpp"
#include "thesauros/types.hpp"
#include "thesauros/utility.hpp"

#include "lineal/base.hpp"
#include "lineal/component-wise/exec-constraint.hpp"
#include "lineal/parallel/index/index.hpp"
#include "lineal/vectorization.hpp"

namespace lineal {
namespace detail {
template<typename T>
struct AdaptivePolicyExecConstraints {
  static constexpr bool supports(const T& /*constraint*/) {
    return false;
  }
};
template<AnyDirectionTag auto tDir>
struct AdaptivePolicyExecConstraints<IterationDirectionConstraint<tDir>> {
  using Con = IterationDirectionConstraint<tDir>;
  static constexpr bool supports(const Con& /*con*/) {
    return true;
  }
  static constexpr bool supports_tiling(const Con& /*con*/) {
    return true;
  }
  static constexpr bool supports_simd(const Con& /*con*/, thes::AnyBoolTag auto /*tiled*/) {
    return true;
  }
};
template<>
struct AdaptivePolicyExecConstraints<ThreadSeqIterConstraint> {
  using Con = ThreadSeqIterConstraint;
  static constexpr bool supports(const Con& /*con*/) {
    return true;
  }
  static constexpr bool supports_tiling(const Con& /*con*/) {
    return false;
  }
  static constexpr bool supports_simd(const Con& /*con*/, thes::AnyBoolTag auto /*tiled*/) {
    return true;
  }
};
template<AnyMatrixNnzPatternTag auto tPattern>
struct AdaptivePolicyExecConstraints<ThreadKeepNbOrderConstraint<tPattern>> {
  using Con = ThreadKeepNbOrderConstraint<tPattern>;
  static constexpr bool supports(const Con& /*con*/) {
    return true;
  }
  static constexpr bool supports_tiling(const Con& /*con*/) {
    return false;
  }
  static constexpr bool supports_simd(const Con& /*con*/, thes::AnyBoolTag auto /*tiled*/) {
    return true;
  }
};
template<>
struct AdaptivePolicyExecConstraints<ThreadKeepNbOrderConstraint<AdjacentStencilNnzPattern{}>> {
  using Con = ThreadKeepNbOrderConstraint<AdjacentStencilNnzPattern{}>;
  static constexpr bool supports(const Con& /*con*/) {
    return true;
  }
  static constexpr bool supports_tiling(const Con& /*con*/) {
    return true;
  }
  static constexpr bool supports_simd(const Con& /*con*/, thes::AnyBoolTag auto /*tiled*/) {
    return true;
  }
};
template<>
struct AdaptivePolicyExecConstraints<GeometryPreservingVectorizationConstraint> {
  using Con = GeometryPreservingVectorizationConstraint;
  static constexpr bool supports(const Con& /*con*/) {
    return true;
  }
  static constexpr bool supports_tiling(const Con& /*con*/) {
    return true;
  }
  static constexpr bool supports_simd(const Con& /*con*/, thes::AnyBoolTag auto tiled) {
    return tiled;
  }
};
} // namespace detail

template<typename TExecutor, typename TTileDims, grex::AnyTag auto tVecTag>
struct AdaptivePolicy {
  using Executor = TExecutor;
  using TileDimsStorage = thes::VoidStorage<TTileDims>;
  using VectorTag = std::decay_t<decltype(tVecTag)>;
  static constexpr auto vector_tag = tVecTag;

  AdaptivePolicy(TExecutor&& exec, std::optional<std::size_t> min_per_thread)
      : ex_{std::forward<TExecutor>(exec)}, min_per_thread_{min_per_thread.value_or(1)} {}
  AdaptivePolicy(std::in_place_t /*tag*/, auto mkexec, std::optional<std::size_t> min_per_thread)
      : ex_{mkexec()}, min_per_thread_{min_per_thread.value_or(1)} {}

  AdaptivePolicy(TExecutor&& exec, TileDimsStorage tile_dims,
                 std::optional<std::size_t> min_per_thread)
      : ex_{std::forward<TExecutor>(exec)}, tile_dims_(tile_dims),
        min_per_thread_{min_per_thread.value_or(1)} {}
  AdaptivePolicy(std::in_place_t /*tag*/, auto mkexec, TileDimsStorage tile_dims,
                 std::optional<std::size_t> min_per_thread)
      : ex_{mkexec()}, tile_dims_(tile_dims), min_per_thread_{min_per_thread.value_or(1)} {}

  template<typename TSize>
  struct ThreadInfo {
    using Indices = thes::IotaRange<TSize>;

    ThreadInfo(std::size_t index, Indices&& indices)
        : index_(index), indices_(std::forward<Indices>(indices)) {}

    [[nodiscard]] std::size_t index() const {
      return index_;
    }

    [[nodiscard]] Indices indices() const {
      return indices_;
    }

  private:
    std::size_t index_;
    Indices indices_;
  };

  template<typename TSink>
  THES_ALWAYS_INLINE constexpr void execute(TSink&& sink,
                                            std::optional<std::size_t> opt_thread_num = {}) const {
    using Sink = std::decay_t<TSink>;
    using Size = Sink::Size;
    using Index = OptGlobalIndex<Sink::is_shared, Size>;

    constexpr auto constraints = lineal::exec_constraints<Sink>;
    static_assert(constraints | thes::star::apply([]<typename... TCons>(const TCons&... con) {
                    return (... && detail::AdaptivePolicyExecConstraints<TCons>::supports(con));
                  }));

    constexpr bool is_tilable =
      !std::is_void_v<TTileDims> &&
      (constraints | thes::star::apply([]<typename... TCons>(const TCons&... con) {
         return (... && detail::AdaptivePolicyExecConstraints<TCons>::supports_tiling(con));
       })) &&
      requires { sink.geometry(); };
    constexpr bool support_simd =
      constraints | thes::star::apply([]<typename... TCons>(const TCons&... con) {
        return (... && detail::AdaptivePolicyExecConstraints<TCons>::supports_simd(
                         con, thes::bool_tag<is_tilable>));
      });

    constexpr bool vectorize = grex::VectorTag<VectorTag> && support_simd && [] {
      if constexpr (is_tilable) {
        using Geometry = std::decay_t<decltype(sink.geometry())>;
        using IdxPos = thes::IndexPosition<Index, std::array<Size, Geometry::dimension_num>>;
        return requires(ThreadInstance<TSink, VectorTag>& tinst, IdxPos idxpos) {
          tinst.compute(idxpos, vector_tag);
        };
      } else {
        return requires(ThreadInstance<TSink, VectorTag>& tinst) {
          tinst.begin().compute(vector_tag);
        };
      }
    }();

    constexpr auto tag = [] {
      if constexpr (vectorize) {
        return vector_tag;
      } else {
        return grex::Scalar{};
      }
    }();

    constexpr auto forward = !(constraints | thes::star::contains(BackwardIterConstraint{}));
    constexpr auto dir = forward ? thes::IterDirection::FORWARD : thes::IterDirection::BACKWARD;

    const Size element_num = sink.size();
    if (element_num == 0) {
      decltype(auto) multi = get_exec_instance<TSink>(std::forward<TSink>(sink), 1);
      ThreadInfo<Size> thread_info{0, thes::range<Size>(0, 0)};
      [[maybe_unused]] decltype(auto) uni = get_thread_instance<TSink>(multi, thread_info, tag);
      return;
    }

    std::size_t thread_num = compute_thread_num(sink, opt_thread_num, element_num,
                                                thes::div_ceil<Size>(element_num, tag.size));
    assert(thread_num <= ex_.thread_num());

    decltype(auto) exec_instance = get_exec_instance<TSink>(std::forward<TSink>(sink), thread_num);

    if constexpr (is_tilable) {
      const auto dim0range = sink.axis_range(thes::index_tag<0>);
      thes::UniformIndexSegmenter seg{
        dim0range.size(),
        std::min<std::size_t>(thread_num, dim0range.size()),
      };
      ex_.execute(
        [&, dim0range, seg](const std::size_t thread_idx) THES_ALWAYS_INLINE {
          if (thread_idx >= seg.segment_num()) {
            return;
          }

          decltype(auto) geometry = sink.geometry();
          assert(!Sink::is_shared || sink.size() == geometry.total_size());
          using Geometry = std::decay_t<decltype(geometry)>;
          const auto factor = geometry.after_size(thes::index_tag<0>);

          const auto range =
            seg.segment_range(thread_idx).transform([&](auto i) THES_ALWAYS_INLINE {
              return i + dim0range.begin_value();
            });
          const Size begin_index = range.begin_value();
          const Size end_index = range.end_value();

          // TODO Here there are global indices while there are own indices in the untiled version!
          ThreadInfo<Size> info{thread_idx, thes::range(begin_index * factor, end_index * factor)};
          decltype(auto) thread_sink = get_thread_instance<TSink>(exec_instance, info, tag);

          auto ranges =
            thes::star::index_transform<Geometry::dimension_num>([&](auto idx) THES_ALWAYS_INLINE {
              if constexpr (idx == 0) {
                return thes::range(begin_index, end_index);
              } else {
                return thes::range(Size{0}, geometry.axis_size(idx));
              }
            });

          if constexpr (vectorize) {
            thes::tiled_for_each<dir>(
              geometry, ranges, tile_dims_, thes::StaticMap{},
              [&](auto idxpos) THES_ALWAYS_INLINE {
                constexpr grex::GeometryRespectingTag<grex::VectorSize<tag.size>> geo_tag{};
                thread_sink.compute(idxpos, geo_tag);
              },
              [&](auto idxpos, auto part) THES_ALWAYS_INLINE {
                const grex::GeometryRespectingTag<grex::VectorPartSize<tag.size>> geo_tag{part};
                thread_sink.compute(idxpos, geo_tag);
              },
              thes::index_tag<tag.size>, thes::type_tag<Index>);
          } else {
            thes::tiled_for_each<dir>(
              geometry, ranges, tile_dims_, thes::StaticMap{},
              [&](auto idxpos) THES_ALWAYS_INLINE { thread_sink.compute(idxpos, tag); },
              thes::type_tag<Index>);
          }
        },
        seg.segment_num());
    } else {
      thes::BlockedIndexSegmenter seg{element_num, std::min<std::size_t>(thread_num, element_num),
                                      Size{tag.size}};
      ex_.execute(
        [&, seg](const std::size_t thread_idx) THES_ALWAYS_INLINE {
          if (thread_idx >= seg.segment_num()) {
            return;
          }

          const Size begin_idx = seg.segment_start(thread_idx);
          const Size end_idx = seg.segment_end(thread_idx);

          ThreadInfo<Size> info(thread_idx, thes::range(begin_idx, end_idx));
          decltype(auto) thread_sink = get_thread_instance<TSink>(exec_instance, info, tag);

          if constexpr (vectorize) {
            const Size end_part = end_idx % Size{tag.size};
            const Size iter_end_index = end_idx - end_part;

            assert(begin_idx <= iter_end_index);

            if constexpr (forward) {
              auto iter = thread_sink.begin() + begin_idx;
              const auto thread_end = thread_sink.begin() + iter_end_index;
              for (; iter != thread_end; iter += tag.size) {
                assert(iter + tag.size - thread_sink.begin() <= element_num);
                iter.compute(tag);
              }
              if (end_part > 0) {
                iter.compute(grex::VectorPartSize<tag.size>{end_part});
              }
            } else {
              auto iter = thread_sink.begin() + iter_end_index;
              const auto thread_begin = thread_sink.begin() + begin_idx;
              if (end_part > 0) {
                iter.compute(grex::VectorPartSize<tag.size>{end_part});
              }
              for (; iter != thread_begin; iter -= tag.size) {
                assert(iter - thread_sink.begin() <= element_num);
                (iter - tag.size).compute(tag);
              }
            }
          } else {
            if constexpr (forward) {
              auto iter = thread_sink.begin() + begin_idx;
              const auto thread_end = thread_sink.begin() + end_idx;
              for (; iter != thread_end; ++iter) {
                iter.compute(tag);
              }
            } else {
              auto iter = thread_sink.begin() + end_idx;
              const auto thread_end = thread_sink.begin() + begin_idx;
              for (; iter != thread_end; --iter) {
                (iter - 1).compute(tag);
              }
            }
          }
        },
        seg.segment_num());
    }
  }

  template<typename TSize>
  void execute_segmented(TSize element_num, auto op,
                         std::optional<std::size_t> opt_thread_num = {}) const {
    if (element_num == 0) {
      return;
    }

    const auto tnum = thes::Optional{opt_thread_num}.value_or_else(
      [&] { return compute_thread_num(element_num, element_num); });
    thes::BlockedIndexSegmenter seg{element_num, tnum, TSize{vector_tag.size}};
    ex_.execute([&, element_num, seg, tnum](const std::size_t thread_idx) THES_ALWAYS_INLINE {
      if (thread_idx < tnum) {
        op(thread_idx, seg.segment_start(thread_idx), seg.segment_end(thread_idx));
      } else {
        op(thread_idx, element_num, element_num);
      }
    });
  }

  [[nodiscard]] std::size_t thread_num() const {
    return ex_.thread_num();
  }

private:
  template<typename TSink>
  using SinkSize = std::decay_t<TSink>::Size;

  template<typename TSink>
  static decltype(auto) get_exec_instance(TSink&& sink, std::size_t thread_num) {
    if constexpr (requires { sink.exec_instance(thread_num); }) {
      return std::forward<TSink>(sink).exec_instance(thread_num);
    } else {
      return std::forward<TSink>(sink);
    }
  };
  template<typename TSink>
  using ExecInstance =
    decltype(get_exec_instance<TSink>(std::declval<TSink>(), std::declval<std::size_t>()));
  template<typename TSink>
  static decltype(auto) get_thread_instance(ExecInstance<TSink>& mtsink,
                                            const ThreadInfo<SinkSize<TSink>>& info,
                                            grex::AnyTag auto tag) {
    if constexpr (requires { mtsink.thread_instance(info, tag); }) {
      return mtsink.thread_instance(info, tag);
    } else {
      return mtsink;
    }
  };
  template<typename TSink, typename TTag>
  using ThreadInstance =
    decltype(get_thread_instance<TSink>(std::declval<ExecInstance<TSink>&>(),
                                        std::declval<ThreadInfo<SinkSize<TSink>>>(),
                                        std::declval<TTag>()));

  template<typename TSink>
  static constexpr auto is_sink_vectorizable(thes::TrueTag /*tile*/) {
    using Sink = std::decay_t<TSink>;
    using Size = Sink::Size;
    using Index = OptGlobalIndex<Sink::is_shared, Size>;
    using Geometry = std::decay_t<decltype(std::declval<TSink>().geometry())>;
    using IdxPos = thes::IndexPosition<Index, std::array<Size, Geometry::dimension_num>>;
    using Tag = grex::GeometryRespectingTag<VectorTag>;
    constexpr Tag tag{vector_tag};

    return requires(ThreadInstance<TSink, Tag>& tinst, IdxPos idxpos) {
      tinst.compute(idxpos, tag);
    };
  }
  template<typename TSink>
  static constexpr auto is_sink_vectorizable(thes::FalseTag /*tile*/) {
    return requires(ThreadInstance<TSink, VectorTag>& tinst) { tinst.begin().compute(vector_tag); };
  }

  template<typename TSize>
  [[nodiscard]] THES_ALWAYS_INLINE std::size_t compute_thread_num(const TSize element_num,
                                                                  const TSize block_num) const {
    using namespace thes::literals;
    const auto thread_min = std::max<std::size_t>(element_num / min_per_thread_, 1_uz);
    const auto block_min = std::min<std::size_t>(thread_min, block_num);
    return std::min(thread_num(), block_min);
  }
  template<typename TSink>
  [[nodiscard]] THES_ALWAYS_INLINE std::size_t
  compute_thread_num(const TSink& sink, std::optional<std::size_t> opt_thread_num,
                     const SinkSize<TSink> element_num, const SinkSize<TSink> block_num) const {
    thes::Optional out{opt_thread_num};
    if constexpr (requires { sink.thread_num(thread_num()); }) {
      out = out.or_else([&]() THES_ALWAYS_INLINE { return sink.thread_num(thread_num()); });
    }
    return out.value_or_else(
      [&]() THES_ALWAYS_INLINE { return compute_thread_num(element_num, block_num); });
  }

  TExecutor ex_;
  TileDimsStorage tile_dims_;
  std::size_t min_per_thread_;
};
} // namespace lineal

#endif // INCLUDE_LINEAL_PARALLEL_EXECUTION_POLICY_ADAPTIVE_HPP
