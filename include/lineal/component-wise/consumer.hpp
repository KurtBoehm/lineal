// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_COMPONENT_WISE_CONSUMER_HPP
#define INCLUDE_LINEAL_COMPONENT_WISE_CONSUMER_HPP

#include <cassert>
#include <cstddef>
#include <type_traits>

#include "grex/tags.hpp"
#include "thesauros/macropolis.hpp"
#include "thesauros/static-ranges.hpp"
#include "thesauros/types.hpp"
#include "thesauros/utility.hpp"

#include "lineal/base/concept.hpp"
#include "lineal/component-wise/exec-constraint.hpp"
#include "lineal/component-wise/facade.hpp"

namespace lineal {
namespace detail {
struct ConsumerSinkConf {
  using Work = void;
  using Value = void;
  static constexpr bool supports_const_access = false;
  static constexpr bool supports_mutable_access = true;
};
} // namespace detail

template<typename TConsumers, SharedVector... TChildren>
struct ConsumerSink;

template<typename... TConsumers, SharedVector... TChildren>
struct ConsumerSink<thes::TypeSeq<TConsumers...>, TChildren...> {
  using Size = thes::Intersection<typename std::decay_t<TChildren>::Size...>;
  using ChildTuple = thes::Tuple<TChildren...>;
  static constexpr auto exec_constraints = merged_exec_constraints<ChildTuple>;
  static constexpr bool is_shared =
    thes::star::unique_value(thes::Tuple{std::decay_t<TChildren>::is_shared...}).value();

  template<typename... TInstances>
  struct ThreadInstance : public facades::ComponentWiseOp<ThreadInstance<TInstances...>,
                                                          detail::ConsumerSinkConf, TChildren&...> {
    using Parent =
      facades::ComponentWiseOp<ThreadInstance, detail::ConsumerSinkConf, TChildren&...>;

    explicit ThreadInstance(TInstances&&... instances, TChildren&... vecs)
        : Parent(vecs...), instances_{std::move(instances)...} {}
    ThreadInstance(const ThreadInstance&) = delete;
    ThreadInstance& operator=(const ThreadInstance&) = delete;
    ThreadInstance(ThreadInstance&&) noexcept = delete;
    ThreadInstance& operator=(ThreadInstance&&) noexcept = delete;
    ~ThreadInstance() = default;

    THES_ALWAYS_INLINE constexpr void compute_iter(grex::AnyTag auto tag, const auto& children,
                                                   auto... iters)
    requires((... && requires { iters.compute(tag); }))
    {
      const thes::Tuple values{iters.compute(tag)...};
      thes::star::static_apply<sizeof...(TInstances)>(
        [&]<std::size_t... tI>() { (..., get<tI>(instances_).compute(tag, children, values)); });
    }
    THES_ALWAYS_INLINE constexpr void compute_base(grex::AnyTag auto tag, const auto& arg,
                                                   const auto& children, auto&... iter_children) {
      const thes::Tuple values{iter_children.compute(arg, tag)...};
      thes::star::static_apply<sizeof...(TInstances)>(
        [&]<std::size_t... tI>() { (..., get<tI>(instances_).compute(tag, children, values)); });
    }

  private:
    thes::Tuple<TInstances...> instances_;
  };

  explicit ConsumerSink(TConsumers&&... consumers, TChildren&&... vecs)
      : consumers_{std::forward<TConsumers>(consumers)...},
        vecs_{std::forward<TChildren>(vecs)...} {}

  auto thread_instance(const auto& info, grex::AnyTag auto tag) {
    return thes::star::static_apply<sizeof...(TConsumers)>([&]<std::size_t... tI>() {
      return thes::star::static_apply<sizeof...(TChildren)>([&]<std::size_t... tJ>() {
        return ThreadInstance<decltype(get<tI>(consumers_).thread_instance(info, tag))...>{
          get<tI>(consumers_).thread_instance(info, tag)...,
          get<tJ>(vecs_)...,
        };
      });
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

  auto value() const {
    return thes::star::static_apply<sizeof...(TConsumers)>(
      [&]<std::size_t... tI>() { return thes::Tuple{get<tI>(consumers_).value()...}; });
  }

private:
  thes::Tuple<TConsumers...> consumers_;
  ChildTuple vecs_;
};
} // namespace lineal

#endif // INCLUDE_LINEAL_COMPONENT_WISE_CONSUMER_HPP
