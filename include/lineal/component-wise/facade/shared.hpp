// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_COMPONENT_WISE_FACADE_SHARED_HPP
#define INCLUDE_LINEAL_COMPONENT_WISE_FACADE_SHARED_HPP

#include <cassert>
#include <compare>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

#include <boost/preprocessor.hpp>

#include "thesauros/iterator.hpp"
#include "thesauros/macropolis.hpp"
#include "thesauros/static-ranges.hpp"
#include "thesauros/types.hpp"
#include "thesauros/utility.hpp"

#include "lineal/base.hpp"
#include "lineal/component-wise/exec-constraint.hpp"
#include "lineal/component-wise/facade/def.hpp"
#include "lineal/component-wise/facade/utility.hpp"
#include "lineal/parallel/distributed-info/def.hpp"
#include "lineal/parallel/index/index.hpp"
#include "lineal/vectorization.hpp"

namespace lineal::facades {
namespace detail {
template<bool tShared, typename TSize, typename TConf>
struct FacadeIdx;
template<typename TSize, typename TConf>
struct FacadeIdx<true, TSize, TConf> {
  using Idx = TSize;
};

template<typename TDerived, typename TConf, OptDistributedInfo TDistInfo>
struct NullDefs {
  using Conf = TConf;
  using Value = Conf::Value;
  using Size = Conf::Size;
  using Diff = std::ptrdiff_t;

  static constexpr bool is_shared = std::is_void_v<TDistInfo>;
  using Idx = FacadeIdx<is_shared, Size, Conf>::Idx;

  static constexpr bool supports_const_access = detail::SupportsConstAccess<Conf>;
  static constexpr bool supports_mutable_access = detail::SupportsMutableAccess<Conf>;

  template<AnyIndex TIdx>
  struct IterProvider {
    using IterBase =
      std::conditional_t<!std::is_void_v<Value>, thes::iter_provider::ValueTypes<Value, Diff>,
                         thes::iter_provider::VoidTypes<Diff>>;
    struct IterTypes : public IterBase {
      using IterState = Size;
    };

    static constexpr Value deref(auto& self) {
      return self.derived_->compute_impl(grex::Scalar{}, TIdx{self.index_});
    }
    template<typename TSelf>
    static thes::TransferConst<TSelf, Size>& state(TSelf& self) {
      return self.index_;
    }
  };

  template<bool tConst, AnyIndex TIdx>
  struct BaseIterator : public thes::IteratorFacade<BaseIterator<tConst, TIdx>,
                                                    thes::iter_provider::Map<IterProvider<TIdx>>> {
    friend struct IterProvider<TIdx>;
    using CDerived = thes::ConditionalConst<tConst, TDerived>;

    BaseIterator(CDerived& self, TIdx index) : derived_(&self), index_(index_value(index)) {}

    THES_ALWAYS_INLINE auto compute(grex::AnyTag auto tag) const
    requires(supports_const_access && tConst && requires { BaseIterator::at_impl(*this, tag); })
    {
      return at_impl(*this, tag);
    }
    THES_ALWAYS_INLINE auto compute(grex::AnyTag auto tag)
    requires(supports_mutable_access && !tConst && requires { BaseIterator::at_impl(*this, tag); })
    {
      return at_impl(*this, tag);
    }

  private:
    THES_ALWAYS_INLINE static auto at_impl(auto& self, grex::AnyTag auto tag)
    requires(requires(TIdx idx) { self.derived_->compute_impl(tag, idx); })
    {
      return self.derived_->compute_impl(tag, TIdx{self.index_});
    }

    CDerived* derived_;
    Size index_;
  };

  template<bool tConst>
  using DefaultIterator = BaseIterator<tConst, Idx>;
};

template<typename TDerived, typename TConf, SharedRange... TChildren>
struct DefaultDefs {
  using Conf = TConf;
  using Value = TConf::Value;
  using Work = TConf::Work;

  static constexpr bool supports_const_access = detail::SupportsConstAccess<TConf>;
  static constexpr bool supports_mutable_access = detail::SupportsMutableAccess<TConf>;
  static constexpr auto component_wise_seq = detail::component_wise_seq<TConf, TChildren...>;
  static constexpr bool supports_store = detail::SupportsStore<TConf>;

  static constexpr bool has_value = !std::is_void_v<Value>;
  static constexpr bool has_real = !std::is_void_v<Work>;

  template<bool tConst>
  using Iterators =
    thes::IndexFilteredTypeSeq<thes::TypeSeq<typename IteratorTrait<TChildren, tConst>::Type...>,
                               component_wise_seq>::AsTuple;
  template<bool tConst>
  using Diff = thes::Intersection<typename std::iterator_traits<
    typename IteratorTrait<TChildren, tConst>::Type>::difference_type...>;

  template<bool tConst>
  struct IterProvider {
    using IterTypes =
      std::conditional_t<has_value, thes::iter_provider::ValueTypes<Value, Diff<tConst>>,
                         thes::iter_provider::VoidTypes<Diff<tConst>>>;
    using IterDiff = Diff<tConst>;

    static constexpr Value deref(auto& self) {
      return self.iterators_ | thes::star::apply([&](auto&... iters) {
               if constexpr (has_real) {
                 return self.derived_->compute_impl(
                   grex::Scalar{}, *self.children_,
                   grex::convert_unsafe<Work>(*iters, grex::Scalar{})...);
               } else {
                 return self.derived_->compute_impl(grex::Scalar{}, *self.children_, iters...);
               }
             });
    }
    static constexpr void incr(auto& self) {
      self.iterators_ | thes::star::for_each([](auto& iter) { ++iter; });
    }
    static constexpr void decr(auto& self) {
      self.iterators_ | thes::star::for_each([](auto& iter) { --iter; });
    }
    static constexpr void iadd(auto& self, auto diff) {
      self.iterators_ | thes::star::for_each([&](auto& iter) { iter += diff; });
    }
    static constexpr void isub(auto& self, auto diff) {
      self.iterators_ | thes::star::for_each([&](auto& iter) { iter -= diff; });
    }
    static constexpr bool eq(const auto& i1, const auto& i2) {
      assert_diff(i1, i2);
      return thes::star::get_at<0>(i1.iterators_) == thes::star::get_at<0>(i2.iterators_);
    }
    static constexpr std::strong_ordering three_way(const auto& i1, const auto& i2) {
      assert_diff(i1, i2);
      return thes::star::get_at<0>(i1.iterators_) <=> thes::star::get_at<0>(i2.iterators_);
    }
    static constexpr IterDiff sub(const auto& i1, const auto& i2) {
      assert_diff(i1, i2);
      return thes::star::get_at<0>(i1.iterators_) - thes::star::get_at<0>(i2.iterators_);
    }

    static constexpr void assert_diff([[maybe_unused]] const auto& i1,
                                      [[maybe_unused]] const auto& i2) {
      assert(thes::star::transform([](const auto& it1, const auto& it2) { return it1 - it2; },
                                   i1.iterators_, i2.iterators_) |
             thes::star::has_unique_value);
    }
  };

public:
  template<bool tConst>
  struct BaseIterator : public thes::IteratorFacade<BaseIterator<tConst>, IterProvider<tConst>> {
    using CDerived = thes::ConditionalConst<tConst, TDerived>;
    using CChildren = thes::ConditionalConst<tConst, thes::Tuple<TChildren...>>;
    using Iterators = DefaultDefs::Iterators<tConst>;

    friend struct IterProvider<tConst>;
    BaseIterator(CDerived& derived, CChildren& children, Iterators iterators)
        : derived_(&derived), children_(&children), iterators_(std::move(iterators)) {}

    THES_ALWAYS_INLINE auto compute(grex::AnyTag auto tag) const
    requires(supports_const_access && tConst && requires { BaseIterator::at_impl(*this, tag); })
    {
      return at_impl(*this, tag);
    }
    THES_ALWAYS_INLINE auto compute(grex::AnyTag auto tag)
    requires(supports_mutable_access && !tConst && requires { BaseIterator::at_impl(*this, tag); })
    {
      return at_impl(*this, tag);
    }
    THES_ALWAYS_INLINE auto store(auto value, grex::AnyTag auto tag)
    requires(supports_store && requires { BaseIterator::store_impl(*this, tag, value); })
    {
      return store_impl(*this, tag, value);
    }

    operator BaseIterator<true>() const {
      return {*derived_, *children_, iterators_};
    }

  private:
    template<grex::AnyTag TTag, typename TIters, bool tHasReal>
    struct HasAtImpl;
    template<grex::AnyTag TTag, typename... TIters>
    struct HasAtImpl<TTag, thes::Tuple<TIters...>, true> {
      static constexpr bool value = requires(CDerived& derived, CChildren& children, TTag tag,
                                             TIters&... iters) {
        derived.compute_impl(tag, children, grex::convert_unsafe<Work>(iters.compute(tag), tag)...);
      };
    };
    template<grex::AnyTag TTag, typename... TIters>
    struct HasAtImpl<TTag, thes::Tuple<TIters...>, false> {
      static constexpr bool value =
        requires(CDerived& derived, CChildren& children, TTag tag, TIters&... iters) {
          derived.compute_impl(tag, children, iters...);
        };
    };
    template<grex::AnyTag TTag>
    THES_ALWAYS_INLINE static auto at_impl(auto& self, TTag tag)
    requires(HasAtImpl<TTag, decltype(self.iterators_), has_real>::value)
    {
      return self.iterators_ | thes::star::apply([&](auto&... iters) THES_ALWAYS_INLINE {
               if constexpr (has_real) {
                 return self.derived_->compute_impl(
                   tag, *self.children_, grex::convert_unsafe<Work>(iters.compute(tag), tag)...);
               } else {
                 return self.derived_->compute_impl(tag, *self.children_, iters...);
               }
             });
    }

    template<grex::AnyTag TTag, typename TValue, typename TIters>
    struct HasStoreImpl;
    template<grex::AnyTag TTag, typename TValue, typename... TIters>
    struct HasStoreImpl<TTag, TValue, thes::Tuple<TIters...>> {
      static constexpr bool value =
        requires(CDerived& derived, CChildren& children, TTag tag, TValue val, TIters&... iters) {
          derived.store_impl(tag, val, children, iters...);
        };
    };
    template<grex::AnyTag TTag, typename TValue>
    THES_ALWAYS_INLINE static auto store_impl(auto& self, TTag tag, TValue value)
    requires(HasStoreImpl<TTag, TValue, decltype(self.iterators_)>::value)
    {
      return self.iterators_ | thes::star::apply([&](auto&... iters) THES_ALWAYS_INLINE {
               return self.derived_->store_impl(tag, value, *self.children_, iters...);
             });
    }

    CDerived* derived_;
    CChildren* children_;
    Iterators iterators_;
  };

  template<bool tConst>
  using DefaultIterator = BaseIterator<tConst>;
};

template<typename TConfDefs>
struct ConstIteratorParent {};
template<typename TConfDefs>
requires(detail::SupportsConstAccess<typename TConfDefs::Conf>)
struct ConstIteratorParent<TConfDefs> {
  using const_iterator = TConfDefs::template DefaultIterator<true>;
};

template<typename TConfDefs>
struct IteratorParent {};
template<typename TConfDefs>
requires(detail::SupportsMutableAccess<typename TConfDefs::Conf>)
struct IteratorParent<TConfDefs> {
  using iterator = TConfDefs::template DefaultIterator<false>;
};

template<typename TConfDefs>
struct AnyIteratorParent : public ConstIteratorParent<TConfDefs>,
                           public IteratorParent<TConfDefs> {};
} // namespace detail

template<typename TDerived, typename TConf, typename TSelfGetter, OptDistributedInfo TDistInfo>
struct SharedNullaryCwOpBase
    : public SharedVectorBase,
      public detail::AnyIteratorParent<detail::NullDefs<TDerived, TConf, TDistInfo>> {
  using Defs = detail::NullDefs<TDerived, TConf, TDistInfo>;
  using RawDistributedInfo = TDistInfo;
  using DistributedInfo = std::decay_t<TDistInfo>;
  using DistributedInfoStorage = thes::VoidStorage<TDistInfo>;

  using Value = Defs::Value;
  using Size = Defs::Size;
  using Diff = Defs::Diff;
  using Index = Defs::Idx;
  static constexpr bool is_shared = Defs::is_shared;
  template<bool tConst, AnyIndex TIdx>
  using BaseIterator = Defs::template BaseIterator<tConst, TIdx>;
  template<bool tConst>
  using DefIterator = Defs::template BaseIterator<tConst, Index>;

  static constexpr bool supports_const_access = detail::SupportsConstAccess<TConf>;
  static constexpr bool supports_mutable_access = detail::SupportsMutableAccess<TConf>;

  static constexpr bool has_value = Defs::has_value;

  explicit SharedNullaryCwOpBase(Size size, TSelfGetter&& self_getter,
                                 thes::VoidStorageRvalRef<TDistInfo> dist_info)
      : size_(size), self_getter_(std::forward<TSelfGetter>(self_getter)),
        dist_info_(std::forward<DistributedInfoStorage>(dist_info)) {}
  explicit SharedNullaryCwOpBase(Size size, TSelfGetter&& self_getter)
  requires(is_shared)
      : size_(size), self_getter_(std::forward<TSelfGetter>(self_getter)), dist_info_{} {}

  explicit SharedNullaryCwOpBase(Size size, thes::VoidStorageRvalRef<TDistInfo> dist_info)
  requires(std::is_default_constructible_v<TSelfGetter>)
      : size_(size), dist_info_(std::forward<DistributedInfoStorage>(dist_info)) {}
  explicit SharedNullaryCwOpBase(Size size)
  requires(is_shared && std::is_default_constructible_v<TSelfGetter>)
      : size_(size), dist_info_{} {}

  [[nodiscard]] Size size() const {
    return size_;
  }

  THES_ALWAYS_INLINE Value operator[](auto index) const
  requires(supports_const_access)
  {
    return at_impl(*this, index, grex::Scalar{});
  }
  THES_ALWAYS_INLINE Value operator[](auto index)
  requires(supports_mutable_access)
  {
    return at_impl(*this, index, grex::Scalar{});
  }

  THES_ALWAYS_INLINE auto compute(const auto& arg, grex::AnyTag auto tag) const
  requires(supports_const_access && requires { SharedNullaryCwOpBase::at_impl(*this, arg, tag); })
  {
    return at_impl(*this, arg, tag);
  }
  THES_ALWAYS_INLINE auto compute(const auto& arg, grex::AnyTag auto tag)
  requires(supports_mutable_access && requires { SharedNullaryCwOpBase::at_impl(*this, arg, tag); })
  {
    return at_impl(*this, arg, tag);
  }

  DefIterator<true> begin() const
  requires(supports_const_access)
  {
    return begin_impl(*this);
  }
  DefIterator<false> begin()
  requires(supports_mutable_access)
  {
    return begin_impl(*this);
  }

  DefIterator<true> end() const
  requires(supports_const_access)
  {
    return end_impl(*this);
  }
  DefIterator<false> end()
  requires(supports_mutable_access)
  {
    return end_impl(*this);
  }

  auto iter_at(auto idx) const
  requires(supports_const_access)
  {
    return iter_at_impl(*this, idx);
  }
  auto iter_at(auto idx)
  requires(supports_mutable_access)
  {
    return iter_at_impl(*this, idx);
  }

  const DistributedInfoStorage& distributed_info_storage() const {
    return dist_info_;
  }
  const DistributedInfoStorage& distributed_info() const
  requires(!is_shared)
  {
    return dist_info_;
  }

private:
  const TDerived& derived() const {
    return self_getter_(*this);
  }
  TDerived& derived() {
    return self_getter_(*this);
  }

  template<typename TSelf>
  THES_ALWAYS_INLINE static DefIterator<std::is_const_v<TSelf>> begin_impl(TSelf& self) {
    return DefIterator<std::is_const_v<TSelf>>{self.derived(), Index{0}};
  }
  template<typename TSelf>
  THES_ALWAYS_INLINE static DefIterator<std::is_const_v<TSelf>> end_impl(TSelf& self) {
    return DefIterator<std::is_const_v<TSelf>>{self.derived(), Index{self.size_}};
  }
  template<typename TSelf, AnyIndex TIdx>
  THES_ALWAYS_INLINE static BaseIterator<std::is_const_v<TSelf>, TIdx> iter_at_impl(TSelf& self,
                                                                                    TIdx idx) {
    return BaseIterator<std::is_const_v<TSelf>, TIdx>{self.derived(), idx};
  }

  THES_ALWAYS_INLINE static auto at_impl(auto& self, const auto& arg, auto tag) {
    return self.derived().compute_impl(tag, arg);
  }

  Size size_;
  [[no_unique_address]] TSelfGetter self_getter_{};
  [[no_unique_address]] DistributedInfoStorage dist_info_;
};
template<typename TDerived, typename TConf, typename TDistInfo = void>
using SharedNullaryCwOp =
  SharedNullaryCwOpBase<TDerived, TConf, detail::DerivedSelfGetter<TDerived>, TDistInfo>;

template<typename TDerived, typename TConf, typename TSelfGetter, SharedRange... TChildren>
struct SharedComponentWiseOpBase
    : public SharedVectorBase,
      public detail::AnyIteratorParent<detail::DefaultDefs<TDerived, TConf, TChildren...>> {
  using Defs = detail::DefaultDefs<TDerived, TConf, TChildren...>;
  using Value = Defs::Value;
  using Work = Defs::Work;

  static constexpr bool supports_const_access = Defs::supports_const_access;
  static constexpr bool supports_mutable_access = Defs::supports_mutable_access;
  static constexpr auto component_wise_seq = Defs::component_wise_seq;
  static constexpr bool custom_range = detail::UsesCustomRange<TConf>;
  static constexpr bool supports_store = Defs::supports_store;
  static constexpr bool is_shared =
    thes::star::unique_value(thes::Tuple{std::decay_t<TChildren>::is_shared...}).value();

  static constexpr auto exec_constraints =
    merged_exec_constraints<decltype(thes::star::only_range<component_wise_seq>(
      std::declval<thes::Tuple<TChildren...>>()))>;

  static constexpr bool has_real = Defs::has_real;

  using Size = thes::Intersection<typename std::decay_t<TChildren>::Size...>;
  using ChildTuple = thes::Tuple<TChildren...>;
  using Children = thes::TypeSeq<std::decay_t<TChildren>...>;

  template<bool tConst>
  using Iterators = Defs::template Iterators<tConst>;
  template<bool tConst>
  using BaseIterator = Defs::template BaseIterator<tConst>;

  explicit SharedComponentWiseOpBase(TChildren&&... children, TSelfGetter self_getter)
      : self_getter_(std::move(self_getter)), children_(std::forward<TChildren>(children)...) {}
  explicit SharedComponentWiseOpBase(TChildren&&... children)
      : children_(std::forward<TChildren>(children)...) {}

  [[nodiscard]] Size size() const {
    if constexpr (custom_range) {
      return derived().size();
    } else {
      assert(children_ | thes::star::transform([](const auto& child) {
               return *thes::safe_cast<Size>(child.size());
             }) |
             thes::star::has_unique_value);
      return *thes::safe_cast<Size>(thes::star::get_at<0>(children_).size());
    }
  }

  THES_ALWAYS_INLINE Value operator[](auto index) const
  requires(supports_const_access)
  {
    return at_impl(*this, index);
  }
  THES_ALWAYS_INLINE decltype(auto) operator[](auto index)
  requires(supports_mutable_access)
  {
    return at_impl(*this, index);
  }

  THES_ALWAYS_INLINE auto compute(const auto& arg, grex::AnyTag auto tag) const
  requires(supports_const_access &&
           requires { SharedComponentWiseOpBase::at_impl(*this, arg, tag); })
  {
    return at_impl(*this, arg, tag);
  }
  THES_ALWAYS_INLINE auto compute(const auto& arg, grex::AnyTag auto tag)
  requires(supports_mutable_access &&
           requires { SharedComponentWiseOpBase::at_impl(*this, arg, tag); })
  {
    return at_impl(*this, arg, tag);
  }
  THES_ALWAYS_INLINE auto store(const auto& arg, auto value, grex::AnyTag auto tag)
  requires(supports_store &&
           requires { SharedComponentWiseOpBase::store_impl(*this, arg, value, tag); })
  {
    return store_impl(*this, arg, value, tag);
  }

  BaseIterator<true> begin() const
  requires(supports_const_access)
  {
    return begin_impl(*this);
  }
  BaseIterator<false> begin()
  requires(supports_mutable_access)
  {
    return begin_impl(*this);
  }

  BaseIterator<true> end() const
  requires(supports_const_access)
  {
    return end_impl(*this);
  }
  BaseIterator<false> end()
  requires(supports_mutable_access)
  {
    return end_impl(*this);
  }

  THES_ALWAYS_INLINE BaseIterator<true> iter_at(const auto& arg) const
  requires(supports_const_access)
  {
    return iter_at_impl(*this, arg);
  }
  THES_ALWAYS_INLINE BaseIterator<false> iter_at(const auto& arg)
  requires(supports_mutable_access)
  {
    return iter_at_impl(*this, arg);
  }

  const ChildTuple& children() const {
    return children_;
  }

  auto axis_range(thes::AnyIndexTag auto idx) const
  requires(requires { impl::axis_range(this->children_, idx); })
  {
    return impl::axis_range(children_, idx);
  }

  decltype(auto) geometry() const
  requires(requires { impl::geometry(this->children_); })
  {
    return impl::geometry(children_);
  }

private:
  THES_ALWAYS_INLINE static decltype(auto) iter_children(auto& self) {
    return thes::star::only_range<component_wise_seq>(self.children_);
  }

#define LINEAL_COMMA_PARAMS(PARAMS) \
  BOOST_PP_REMOVE_PARENS(BOOST_PP_IF(BOOST_PP_CHECK_EMPTY PARAMS, (), (, ))) \
  BOOST_PP_REMOVE_PARENS(PARAMS)

#define LINEAL_ITER_IMPL(NAME, DECL_PARAMS, PARAMS) \
  template<typename TSelf> \
  THES_ALWAYS_INLINE static BaseIterator<std::is_const_v<TSelf>> NAME##_impl( \
    TSelf& self LINEAL_COMMA_PARAMS(DECL_PARAMS)) { \
    constexpr bool is_const = std::is_const_v<TSelf>; \
    using Iter = BaseIterator<is_const>; \
    using Iters = Iterators<is_const>; \
\
    return iter_children(self) | thes::star::enumerate<std::size_t> | \
           thes::star::apply([&](auto... children) { \
             if constexpr (custom_range) { \
               return Iter{self.derived(), self.children_, \
                           Iters{self.derived().NAME##_impl(thes::star::get_at<0>(children), \
                                                            thes::star::get_at<1>(children) \
                                                              LINEAL_COMMA_PARAMS(PARAMS))...}}; \
             } else { \
               return Iter{ \
                 self.derived(), self.children_, \
                 Iters{thes::star::get_at<1>(children).NAME(BOOST_PP_REMOVE_PARENS(PARAMS))...}}; \
             } \
           }); \
  }

  LINEAL_ITER_IMPL(begin, (), ())
  LINEAL_ITER_IMPL(end, (), ())
  LINEAL_ITER_IMPL(iter_at, (const auto& arg), (arg))

#undef LINEAL_ITER_IMPL
#undef LINEAL_COMMA_PARAMS

  template<typename TSelf, typename TIdx, typename TOffspring, bool tHasReal>
  struct HasAtImplScalar;
  template<typename TSelf, typename TIdx, typename... TOffspring>
  struct HasAtImplScalar<TSelf, TIdx, thes::Tuple<TOffspring...>, true> {
    static constexpr bool value = requires(TSelf& self, TIdx index, TOffspring&... children) {
      self.derived().compute_impl(grex::Scalar{}, self.children_,
                                  grex::convert_unsafe<Work>(children[index], grex::Scalar{})...);
    };
  };
  template<typename TSelf, typename TIdx, typename... TOffspring>
  struct HasAtImplScalar<TSelf, TIdx, thes::Tuple<TOffspring...>, false> {
    static constexpr bool value = requires(TSelf& self, TIdx index, TOffspring&... children) {
      self.derived().compute_impl(grex::Scalar{}, index, self.children_, children...);
    };
  };
  template<typename TSelf, typename TIdx>
  THES_ALWAYS_INLINE static decltype(auto) at_impl(TSelf& self, TIdx index)
  requires(HasAtImplScalar<TSelf, TIdx, decltype(thes::star::to_tuple(iter_children(self))),
                           has_real>::value)
  {
    return iter_children(self) |
           thes::star::apply([&](auto&... children) THES_ALWAYS_INLINE -> decltype(auto) {
             if constexpr (has_real) {
               return self.derived().compute_impl(
                 grex::Scalar{}, self.children_,
                 grex::convert_unsafe<Work>(children[index], grex::Scalar{})...);
             } else {
               return self.derived().compute_impl(grex::Scalar{}, index, self.children_,
                                                  children...);
             }
           });
  }

  template<typename TSelf, typename TArg, grex::AnyTag TTag, typename TOffspring, bool tHasReal>
  struct HasAtImplVector;
  template<typename TSelf, typename TArg, grex::AnyTag TTag, typename... TOffspring>
  struct HasAtImplVector<TSelf, TArg, TTag, thes::Tuple<TOffspring...>, true> {
    static constexpr bool value =
      requires(TSelf& self, TArg arg, TTag tag, TOffspring&... children) {
        self.derived().compute_impl(tag, self.children_,
                                    grex::convert_unsafe<Work>(children.compute(arg, tag), tag)...);
      };
  };
  template<typename TSelf, typename TArg, grex::AnyTag TTag, typename... TOffspring>
  struct HasAtImplVector<TSelf, TArg, TTag, thes::Tuple<TOffspring...>, false> {
    static constexpr bool value =
      requires(TSelf& self, TArg arg, TTag tag, TOffspring&... children) {
        self.derived().compute_impl(tag, arg, self.children_, children...);
      };
  };
  template<typename TSelf, typename TArg, grex::AnyTag TTag>
  THES_ALWAYS_INLINE static decltype(auto) at_impl(TSelf& self, const TArg& arg, TTag tag)
  requires(HasAtImplVector<TSelf, TArg, TTag, decltype(thes::star::to_tuple(iter_children(self))),
                           has_real>::value)
  {
    return iter_children(self) |
           thes::star::apply([&](auto&... children) THES_ALWAYS_INLINE -> decltype(auto) {
             if constexpr (has_real) {
               return self.derived().compute_impl(
                 tag, self.children_,
                 grex::convert_unsafe<Work>(children.compute(arg, tag), tag)...);
             } else {
               return self.derived().compute_impl(tag, arg, self.children_, children...);
             }
           });
  }

  template<typename TSelf, typename TArg, typename TValue, grex::AnyTag TTag, typename TOffspring>
  struct HasStoreImpl;
  template<typename TSelf, typename TArg, typename TValue, grex::AnyTag TTag,
           typename... TOffspring>
  struct HasStoreImpl<TSelf, TArg, TValue, TTag, thes::Tuple<TOffspring...>> {
    static constexpr bool value =
      requires(TSelf& self, const TArg& arg, TValue val, TTag tag, TOffspring&... children) {
        self.derived().store_impl(tag, arg, val, self.children_, children...);
      };
  };
  template<typename TSelf, typename TArg, typename TValue, grex::AnyTag TTag>
  THES_ALWAYS_INLINE static auto store_impl(TSelf& self, const TArg& arg, TValue value, TTag tag)
  requires(HasStoreImpl<TSelf, TArg, TValue, TTag,
                        decltype(thes::star::to_tuple(iter_children(self)))>::value)
  {
    return iter_children(self) |
           thes::star::apply([&](auto&... children) THES_ALWAYS_INLINE -> decltype(auto) {
             return self.derived().store_impl(tag, arg, value, self.children_, children...);
           });
  }

  const auto& derived() const {
    return self_getter_(*this);
  }
  auto& derived() {
    return self_getter_(*this);
  }

  [[no_unique_address]] TSelfGetter self_getter_{};
  ChildTuple children_;
};

template<typename TDerived, typename TConf, SharedRange... TChildren>
using SharedComponentWiseOp =
  SharedComponentWiseOpBase<TDerived, TConf, detail::DerivedSelfGetter<TDerived>, TChildren...>;
} // namespace lineal::facades

#endif // INCLUDE_LINEAL_COMPONENT_WISE_FACADE_SHARED_HPP
