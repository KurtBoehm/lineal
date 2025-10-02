// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_COMPONENT_WISE_SINK_ACCUMULATOR_HPP
#define INCLUDE_LINEAL_COMPONENT_WISE_SINK_ACCUMULATOR_HPP

#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "thesauros/macropolis.hpp"
#include "thesauros/static-ranges.hpp"
#include "thesauros/types.hpp"
#include "thesauros/utility.hpp"

#include "lineal/base.hpp"
#include "lineal/component-wise/consumer.hpp"
#include "lineal/parallel/distributed-info.hpp"
#include "lineal/vectorization.hpp"

namespace lineal {
template<typename TReal, typename TComponentOp, typename TMergeOp, typename TPostOp, typename TIdxs>
struct AccumulationConsumer;
template<typename TReal, typename TComponentOp, typename TMergeOp, typename TPostOp,
         std::size_t... tIdxs>
struct AccumulationConsumer<TReal, TComponentOp, TMergeOp, TPostOp, std::index_sequence<tIdxs...>> {
  using Real = TReal;
  static constexpr std::array<std::size_t, sizeof...(tIdxs)> indices{tIdxs...};

  template<grex::AnyTag TVecTag>
  struct ThreadInstance {
    using Numeric = grex::TagType<TVecTag, TReal>;

    ThreadInstance(TComponentOp component, TMergeOp merge, TReal& sum)
        : component_op_(std::move(component)), merge_op_(std::move(merge)), sum_(sum) {}
    ThreadInstance(const ThreadInstance&) = delete;
    ThreadInstance& operator=(const ThreadInstance&) = delete;
    ThreadInstance(ThreadInstance&&) noexcept = default;
    ThreadInstance& operator=(ThreadInstance&&) noexcept = delete;
    ~ThreadInstance() {
      merge_op_(std::atomic_ref{sum_}, thread_sum_, TVecTag{});
      // std::atomic_ref{sum_} += grex::horizontal_add(thread_sum_, TVecTag{});
    }

    THES_ALWAYS_INLINE constexpr void compute(auto tag, const auto& /*children*/, auto values) {
      component_op_(tag, thread_sum_, compat::cast<TReal>(get<tIdxs>(values))...);
      // thread_sum_ += op_(values..., tag);
    }

  private:
    [[no_unique_address]] TComponentOp component_op_;
    [[no_unique_address]] TMergeOp merge_op_;
    Numeric thread_sum_{0};
    TReal& sum_;
  };

  AccumulationConsumer(TComponentOp component, TMergeOp merge, TPostOp post)
      : component_op_(std::move(component)), merge_op_(std::move(merge)),
        post_op_(std::move(post)) {}

  template<grex::AnyTag TVecTag>
  ThreadInstance<TVecTag> thread_instance(const auto& /*info*/, TVecTag /*tag*/) {
    return {component_op_, merge_op_, sum_};
  }

  TReal sum() const {
    std::atomic_thread_fence(std::memory_order::seq_cst);
    return sum_;
  }
  TReal value() const {
    return post_op_(sum());
  }

private:
  [[no_unique_address]] TComponentOp component_op_;
  [[no_unique_address]] TMergeOp merge_op_;
  [[no_unique_address]] TPostOp post_op_;
  TReal sum_{0};
};

namespace detail {
template<typename TReal, std::size_t... tIdxs, typename TComponentOp, typename TMergeOp,
         typename TPostOp>
THES_ALWAYS_INLINE inline AccumulationConsumer<TReal, TComponentOp, TMergeOp, TPostOp,
                                               std::index_sequence<tIdxs...>>
accumulation_consumer(TComponentOp component, TMergeOp merge, TPostOp post,
                      std::index_sequence<tIdxs...> /*idxs*/ = {}) {
  return {std::move(component), std::move(merge), std::move(post)};
}

template<typename TConsumer, SharedVector... TVecs>
constexpr std::decay_t<TConsumer>::Real accumulate_full(TConsumer&& consumer, const auto& expo,
                                                        TVecs&&... vecs) {
  using Sink = ConsumerSink<thes::TypeSeq<TConsumer>, TVecs...>;
  Sink sink{std::forward<TConsumer>(consumer), std::forward<TVecs>(vecs)...};
  expo.execute(sink);
  const auto [value] = sink.value();
  return value;
}

template<typename TReal, std::size_t... tIdxs, typename TOp, typename TPostOp, typename TDistInfo>
constexpr auto accumulate_consumer(std::index_sequence<tIdxs...> idxs, TOp /*op*/, TPostOp post,
                                   const TDistInfo& /*dist_info*/) {
  auto component = [](auto tag, auto& thread_sum, auto... values) {
    thread_sum += TOp{}(values..., tag);
  };
  auto merge = [](auto&& sum, auto thread_sum, auto tag) {
    sum += grex::horizontal_add(thread_sum, tag);
  };
  auto wpost = [&](TReal value) { return post(value); };
  return accumulation_consumer<TReal>(component, merge, wpost, idxs);
}

template<typename TReal, typename TOp, typename TPostOp, AnyVector... TVecs>
constexpr TReal accumulate(TOp op, TPostOp post, const auto& expo, TVecs&&... vecs) {
  return accumulate_full(accumulate_consumer<TReal>(std::index_sequence_for<TVecs...>{},
                                                    std::move(op), std::move(post),
                                                    unique_distributed_info_storage(vecs...)),
                         expo, std::forward<TVecs>(vecs)...);
}

template<typename TReal>
inline constexpr auto euclidean_impl =
  [](auto value, auto tag) { return tag.mask(compat::euclidean_squared<TReal>(value)); };
} // namespace detail

template<typename TReal, std::size_t tI, typename TDistInfo = thes::Empty>
constexpr auto euclidean_squared_consumer(TDistInfo&& dist_info = {}) {
  return detail::accumulate_consumer<TReal>(
    std::index_sequence<tI>{}, detail::euclidean_impl<TReal>, [](TReal v) { return v; },
    std::forward<TDistInfo>(dist_info));
}
template<typename TReal, std::size_t tI, typename TDistInfo = thes::Empty>
constexpr auto euclidean_norm_consumer(TDistInfo&& dist_info = {}) {
  return detail::accumulate_consumer<TReal>(
    std::index_sequence<tI>{}, detail::euclidean_impl<TReal>, [](TReal v) { return grex::sqrt(v); },
    std::forward<TDistInfo>(dist_info));
}
template<typename TReal, std::size_t tI, typename TDistInfo = thes::Empty>
constexpr auto inv_euclidean_norm_consumer(TDistInfo&& dist_info = {}) {
  return detail::accumulate_consumer<TReal>(
    std::index_sequence<tI>{}, detail::euclidean_impl<TReal>,
    [](TReal v) { return thes::fast::rsqrt(v); }, std::forward<TDistInfo>(dist_info));
}
template<typename TReal, std::size_t tI, std::size_t tJ, typename TDistInfo = thes::Empty>
constexpr auto dot_consumer(TDistInfo&& dist_info = {}) {
  return detail::accumulate_consumer<TReal>(
    std::index_sequence<tI, tJ>{},
    /*component=*/
    [](auto val1, auto val2, auto tag) { return tag.mask(compat::dot<TReal>(val1, val2)); },
    /*post=*/[](TReal v) { return v; }, std::forward<TDistInfo>(dist_info));
}
template<typename TReal, std::size_t tI, typename TDistInfo = thes::Empty>
constexpr auto max_norm_consumer(const TDistInfo& dist_info = {}) {
  auto component = [](grex::AnyTag auto tag, auto& thread_sum, auto value) {
    thread_sum = grex::max(thread_sum, tag.mask(compat::max_norm<TReal>(value)));
  };
  auto merge = [](auto&& sum, auto thread_sum, auto tag) {
    const auto max = grex::horizontal_max(thread_sum, tag);
    auto expected = sum.load();
    while (!sum.compare_exchange_weak(expected, std::max(expected, max))) {
    }
  };
  auto post = [&](TReal v) { return v; };
  return detail::accumulation_consumer<TReal>(component, merge, post, std::index_sequence<tI>{});
}

template<typename TReal, AnyVector TVec>
constexpr TReal euclidean_squared(TVec&& vec, const auto& expo) {
  decltype(auto) dinfo = distributed_info_storage(vec);
  return detail::accumulate_full(euclidean_squared_consumer<TReal, 0>(dinfo), expo,
                                 std::forward<TVec>(vec));
}
template<typename TReal, AnyVector TVec>
constexpr TReal euclidean_norm(TVec&& vec, const auto& expo) {
  decltype(auto) dinfo = distributed_info_storage(vec);
  return detail::accumulate_full(euclidean_norm_consumer<TReal, 0>(dinfo), expo,
                                 std::forward<TVec>(vec));
}
template<typename TReal, AnyVector TVec>
constexpr TReal inv_euclidean_norm(TVec&& vec, const auto& expo) {
  decltype(auto) dinfo = distributed_info_storage(vec);
  return detail::accumulate_full(inv_euclidean_norm_consumer<TReal, 0>(dinfo), expo,
                                 std::forward<TVec>(vec));
}
template<typename TReal, AnyVector TVec1, AnyVector TVec2>
constexpr TReal dot(TVec1&& vec1, TVec2&& vec2, const auto& expo) {
  decltype(auto) dinfo = unique_distributed_info_storage(vec1, vec2);
  return detail::accumulate_full(dot_consumer<TReal, 0, 1>(dinfo), expo, std::forward<TVec1>(vec1),
                                 std::forward<TVec2>(vec2));
}
template<typename TReal, AnyVector TVec>
constexpr TReal max_norm(TVec&& vec, const auto& expo) {
  decltype(auto) dinfo = distributed_info_storage(vec);
  return detail::accumulate_full(max_norm_consumer<TReal, 0>(dinfo), expo, std::forward<TVec>(vec));
};

template<typename... TConsumers, SharedVector... TVecs>
constexpr auto multiconsume(thes::Tuple<TConsumers...> consumers, thes::Tuple<TVecs...> vecs,
                            const auto& expo) {
  using Sink = ConsumerSink<thes::TypeSeq<TConsumers...>, TVecs...>;
  Sink sink = thes::star::static_apply<sizeof...(TConsumers)>([&]<std::size_t... tI>() {
    return thes::star::static_apply<sizeof...(TVecs)>([&]<std::size_t... tJ>() {
      return Sink{
        std::forward<TConsumers>(get<tI>(consumers))...,
        std::forward<TVecs>(get<tJ>(vecs))...,
      };
    });
  });
  expo.execute(sink);
  return sink.value();
}
} // namespace lineal

#endif // INCLUDE_LINEAL_COMPONENT_WISE_SINK_ACCUMULATOR_HPP
