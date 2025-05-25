// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_COMPONENT_WISE_SINK_ACCUMULATOR_HPP
#define INCLUDE_LINEAL_COMPONENT_WISE_SINK_ACCUMULATOR_HPP

#include <atomic>
#include <cassert>
#include <type_traits>
#include <utility>

#include "thesauros/macropolis.hpp"
#include "thesauros/math.hpp"
#include "thesauros/static-ranges.hpp"
#include "thesauros/types.hpp"
#include "thesauros/utility.hpp"

#include "lineal/base.hpp"
#include "lineal/component-wise/facade.hpp"
#include "lineal/parallel/def/communicator.hpp"
#include "lineal/vectorization.hpp"

namespace lineal {
namespace detail {
template<typename TReal>
struct AccumulatorSinkConf {
  using Work = TReal;
  using Value = void;
  static constexpr bool supports_const_access = false;
  static constexpr bool supports_mutable_access = true;
};
} // namespace detail

template<typename TReal, typename TComponentOp, typename TMergeOp, SharedVector... TChildren>
struct AccumulatorSink {
  using Size = thes::Intersection<typename std::decay_t<TChildren>::Size...>;
  using ChildTuple = thes::Tuple<TChildren...>;
  static constexpr auto exec_constraints = merged_exec_constraints<ChildTuple>;
  static constexpr bool is_shared =
    thes::star::unique_value(thes::Tuple{std::decay_t<TChildren>::is_shared...}).value();

  template<grex::AnyTag TVecTag>
  struct ThreadInstance
      : public facades::ComponentWiseOp<ThreadInstance<TVecTag>, detail::AccumulatorSinkConf<TReal>,
                                        TChildren&...> {
    using Numeric = grex::TagType<TReal, TVecTag>;
    using Parent =
      facades::ComponentWiseOp<ThreadInstance, detail::AccumulatorSinkConf<TReal>, TChildren&...>;

    explicit ThreadInstance(TChildren&... vecs, TComponentOp component, TMergeOp merge, TReal& sum)
        : Parent(vecs...), component_op_(std::move(component)), merge_op_(std::move(merge)),
          sum_(sum) {}
    ThreadInstance(const ThreadInstance&) = delete;
    ThreadInstance& operator=(const ThreadInstance&) = delete;
    ThreadInstance(ThreadInstance&&) noexcept = delete;
    ThreadInstance& operator=(ThreadInstance&&) noexcept = delete;
    ~ThreadInstance() {
      merge_op_(std::atomic_ref{sum_}, thread_sum_, TVecTag{});
      // std::atomic_ref{sum_} += grex::horizontal_add(thread_sum_, TVecTag{});
    }

    THES_ALWAYS_INLINE constexpr void
    compute_impl(auto tag, const auto& /*children*/,
                 std::conditional_t<true, Numeric, TChildren>... values) {
      component_op_(tag, thread_sum_, values...);
      // thread_sum_ += op_(values..., tag);
    }

  private:
    TComponentOp component_op_;
    TMergeOp merge_op_;
    Numeric thread_sum_{0};
    TReal& sum_;
  };

  explicit AccumulatorSink(TChildren&&... vecs, TComponentOp component, TMergeOp merge)
      : vecs_{std::forward<TChildren>(vecs)...}, component_op_(std::move(component)),
        merge_op_(std::move(merge)) {}

  template<grex::AnyTag TVecTag>
  ThreadInstance<TVecTag> thread_instance(const auto& /*info*/, TVecTag /*tag*/) {
    return vecs_ | thes::star::apply([&](TChildren&... children) {
             return ThreadInstance<TVecTag>{children..., component_op_, merge_op_, sum_};
           });
  }

  [[nodiscard]] Size size() const {
    assert(vecs_ | thes::star::transform([](const auto& child) { return child.size(); }) |
           thes::star::has_unique_value);
    return *thes::safe_cast<Size>(thes::star::get_at<0>(vecs_).size());
  }
  auto axis_range(thes::AnyIndexTag auto idx) const
  requires(requires(const ChildTuple& vecs) { impl::axis_range(vecs, idx); })
  {
    return impl::axis_range(vecs_, idx);
  }
  decltype(auto) geometry() const
  requires(requires(const ChildTuple& vecs) { impl::geometry(vecs); })
  {
    return impl::geometry(vecs_);
  }

  TReal sum() const {
    std::atomic_thread_fence(std::memory_order::seq_cst);
    return sum_;
  }

private:
  ChildTuple vecs_;
  TComponentOp component_op_;
  TMergeOp merge_op_;
  TReal sum_{0};
};

namespace detail {
template<typename TReal, typename TComponentOp, typename TMergeOp, SharedVector... TVecs>
inline constexpr TReal accumulate_full(TComponentOp component, TMergeOp merge, const auto& expo,
                                       TVecs&&... vecs) {
  using Sink = AccumulatorSink<TReal, TComponentOp, TMergeOp, TVecs...>;
  auto sink = Sink(std::forward<TVecs>(vecs)..., std::move(component), std::move(merge));
  expo.execute(sink);
  return sink.sum();
};

template<typename TOp>
inline constexpr auto accumulate_reduce =
  [](auto tag, auto& thread_sum, auto... values) { thread_sum += TOp{}(values..., tag); };
inline constexpr auto accumulate_combine = [](auto&& sum, auto thread_sum, auto tag) {
  sum += grex::horizontal_add(thread_sum, tag);
};

template<typename TReal, typename TOp, SharedVector... TVecs>
inline constexpr TReal accumulate(TOp /*op*/, const auto& expo, TVecs&&... vecs) {
  return accumulate_full<TReal>(accumulate_reduce<TOp>, accumulate_combine, expo,
                                std::forward<TVecs>(vecs)...);
};

inline constexpr auto euclidean_impl = [](auto value, auto tag) { return tag.mask(value * value); };
inline constexpr auto dot_impl = [](auto val1, auto val2, auto tag) {
  return tag.mask(val1 * val2);
};
} // namespace detail

template<typename TReal, AnyVector TVec>
inline constexpr TReal euclidean_squared(TVec&& vec, const auto& expo) {
  return detail::accumulate<TReal>(detail::euclidean_impl, expo, std::forward<TVec>(vec));
}
template<typename TReal, AnyVector TVec>
inline constexpr TReal euclidean_norm(TVec&& vec, const auto& expo) {
  return thes::fast::sqrt(euclidean_squared<TReal>(std::forward<TVec>(vec), expo));
}
template<typename TReal, AnyVector TVec>
inline constexpr TReal inv_euclidean_norm(TVec&& vec, const auto& expo) {
  return thes::fast::rsqrt(euclidean_squared<TReal>(std::forward<TVec>(vec), expo));
}

template<typename TReal, AnyVector TVec1, AnyVector TVec2>
inline constexpr TReal dot(TVec1&& vec1, TVec2&& vec2, const auto& expo) {
  return detail::accumulate<TReal>(detail::dot_impl, expo, std::forward<TVec1>(vec1),
                                   std::forward<TVec2>(vec2));
}

namespace detail {
inline constexpr auto max_norm_reduce = [](auto tag, auto& thread_sum, auto value) {
  thread_sum = grex::max(thread_sum, grex::abs(value, tag), tag);
};
inline constexpr auto max_norm_combine = [](auto&& sum, auto thread_sum, auto tag) {
  const auto max = grex::horizontal_max(thread_sum, tag);
  auto expected = sum.load();
  while (!sum.compare_exchange_weak(expected, std::max(expected, max))) {
  }
};
} // namespace detail

template<typename TReal, SharedVector TVec>
inline constexpr TReal max_norm(TVec&& vec, const auto& expo) {
  return detail::accumulate_full<TReal>(detail::max_norm_reduce, detail::max_norm_combine, expo,
                                        std::forward<TVec>(vec));
};
} // namespace lineal

#endif // INCLUDE_LINEAL_COMPONENT_WISE_SINK_ACCUMULATOR_HPP
