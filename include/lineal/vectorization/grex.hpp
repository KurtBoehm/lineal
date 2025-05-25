// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_VECTORIZATION_GREX_HPP
#define INCLUDE_LINEAL_VECTORIZATION_GREX_HPP

#include <bit>
#include <climits>
#include <concepts>
#include <cstddef>
#include <span>
#include <type_traits>
#include <utility>

#include "stado/stado.hpp" // IWYU pragma: export
#include "thesauros/algorithms.hpp"
#include "thesauros/macropolis.hpp"
#include "thesauros/static-ranges.hpp"
#include "thesauros/types.hpp"

namespace grex {
inline constexpr std::size_t max_vector_bytes =
  sizeof(thes::f32) * stado::native_sizes<thes::f32>.back();

using stado::AnyMask;
using stado::AnyVector;
using stado::SizeMask;
using stado::SizeVector;
using stado::TypeVector;

// TODO f16
template<typename T>
concept Vectorizable =
  std::same_as<T, thes::i8> || std::same_as<T, thes::u8> || std::same_as<T, thes::i16> ||
  std::same_as<T, thes::u16> || std::same_as<T, thes::i32> || std::same_as<T, thes::u32> ||
  std::same_as<T, thes::i64> || std::same_as<T, thes::u64> || std::same_as<T, thes::f32> ||
  std::same_as<T, thes::f64>;
template<Vectorizable T>
inline constexpr std::size_t max_vector_size = max_vector_bytes / sizeof(T);

using stado::Vector;

template<typename T, std::size_t tElementNum>
requires(std::has_single_bit(tElementNum))
using Mask = stado::Mask<CHAR_BIT * sizeof(T), tElementNum>;

template<typename T>
inline constexpr auto native_sizes = stado::native_sizes<T>;
template<AnyVector TVec>
using MaskFor = grex::Mask<typename TVec::Value, TVec::size>;

template<typename T>
struct FloatSizeTrait {
  using Type = T;
};
template<>
struct FloatSizeTrait<thes::f32> {
  using Type = thes::u32;
};
template<>
struct FloatSizeTrait<thes::f64> {
  using Type = thes::u64;
};
template<typename T>
using FloatSize = FloatSizeTrait<T>::Type;

template<typename T>
struct TypedScalar {
  [[nodiscard]] constexpr T mask(T value) const {
    return value;
  }
};
struct Scalar {
  using Full = Scalar;
  static constexpr std::size_t size = 1;

  template<typename T>
  [[nodiscard]] TypedScalar<T> instantiate(thes::TypeTag<T> /*tag*/) {
    return {};
  }

  template<typename T>
  [[nodiscard]] constexpr T mask(T value) const {
    return value;
  }
  auto cast(thes::AnyTypeTag auto /*tag*/) const {
    return Scalar{};
  }
};

template<std::size_t tSize>
struct VectorSize;

template<typename T, std::size_t tSize>
struct TypedVectorSize {
  using Full = VectorSize<tSize>;
  using Type = T;
  static constexpr std::size_t size = tSize;

  using Mask = grex::Mask<Type, size>;

  [[nodiscard]] Mask mask() const {
    return Mask(true);
  }
  template<SizeVector<size> TVec>
  [[nodiscard]] TVec mask(TVec vector) const {
    return vector;
  }
  [[nodiscard]] Mask mask(Mask mask) const {
    return mask;
  }

  template<typename TOther>
  requires(sizeof(T) == sizeof(TOther))
  auto cast(thes::TypeTag<TOther> /*tag*/) const {
    return TypedVectorSize<TOther, tSize>{};
  }

  [[nodiscard]] std::size_t part() const {
    return size;
  }
};

template<std::size_t tSize>
struct VectorSize {
  using Full = VectorSize<tSize>;
  static constexpr std::size_t size = tSize;

  template<typename T>
  [[nodiscard]] TypedVectorSize<T, size> instantiate(thes::TypeTag<T> /*tag*/) {
    return {};
  }

  template<SizeVector<size> TVec>
  [[nodiscard]] TVec mask(TVec vector) const {
    return vector;
  }
  template<SizeMask<size> TMask>
  [[nodiscard]] TMask mask(TMask mask) const {
    return mask;
  }

  [[nodiscard]] std::size_t part() const {
    return size;
  }
};

template<typename T, std::size_t tSize>
struct TypedVectorMaskSize {
  using Type = T;
  using Full = VectorSize<tSize>;
  static constexpr std::size_t size = tSize;

  using Mask = grex::Mask<Type, size>;

  explicit TypedVectorMaskSize(Mask mask) : mask_(mask) {}

  [[nodiscard]] Mask mask() const {
    return mask_;
  }

  template<SizeVector<size> TVec>
  [[nodiscard]] TVec mask(TVec vector) const {
    return stado::select(mask_, vector, TVec(0));
  }

  [[nodiscard]] Mask mask(Mask mask) const {
    return mask_ & mask;
  }

  template<typename TOther>
  requires(sizeof(T) == sizeof(TOther))
  auto cast(thes::TypeTag<TOther> /*tag*/) const {
    return TypedVectorMaskSize<TOther, tSize>{mask_};
  }

private:
  Mask mask_;
};

template<std::size_t tSize>
struct VectorPartSize {
  using Full = VectorSize<tSize>;
  static constexpr std::size_t size = tSize;

  explicit constexpr VectorPartSize(std::size_t part) : part_(part) {}

  template<typename T>
  [[nodiscard]] TypedVectorMaskSize<T, size> instantiate(thes::TypeTag<T> /*tag*/) {
    using Mask = Mask<T, size>;
    return TypedVectorMaskSize<T, size>{make_mask<Mask>()};
  }

  template<typename TMask>
  auto make_mask(thes::TypeTag<TMask> /*tag*/ = {}) const {
    return stado::part_mask<TMask>(part_);
  }
  template<SizeMask<size> TMask>
  [[nodiscard]] TMask mask(TMask mask) const {
    return mask & make_mask<TMask>();
  }
  template<SizeVector<size> TVec>
  [[nodiscard]] TVec mask(TVec vector) const {
    vector.cutoff(part_);
    return vector;
  }

  [[nodiscard]] std::size_t part() const {
    return part_;
  }

private:
  std::size_t part_;
};

template<typename TTag>
struct GeometryRespectingTag : public TTag {
  static constexpr bool is_geometry_respecting = true;

  using TTag::TTag;
  explicit constexpr GeometryRespectingTag(TTag tag) : TTag(tag) {}
};

// tag types

template<typename TTag>
struct IsTagTrait {
  static constexpr bool is_tag = false;
  static constexpr bool is_vector_tag = false;
  static constexpr bool is_full_vector_tag = false;
};
template<>
struct IsTagTrait<Scalar> {
  static constexpr bool is_tag = true;
  static constexpr bool is_vector_tag = false;
  static constexpr bool is_full_vector_tag = false;
};
template<typename T>
struct IsTagTrait<TypedScalar<T>> {
  static constexpr bool is_tag = true;
  static constexpr bool is_vector_tag = false;
  static constexpr bool is_full_vector_tag = false;
};
template<std::size_t tSize>
struct IsTagTrait<VectorSize<tSize>> {
  static constexpr bool is_tag = true;
  static constexpr bool is_vector_tag = true;
  static constexpr bool is_full_vector_tag = true;
};
template<typename T, std::size_t tSize>
struct IsTagTrait<TypedVectorSize<T, tSize>> {
  static constexpr bool is_tag = true;
  static constexpr bool is_vector_tag = true;
  static constexpr bool is_full_vector_tag = true;
};
template<std::size_t tSize>
struct IsTagTrait<VectorPartSize<tSize>> {
  static constexpr bool is_tag = true;
  static constexpr bool is_vector_tag = true;
  static constexpr bool is_full_vector_tag = false;
};
template<typename T, std::size_t tSize>
struct IsTagTrait<TypedVectorMaskSize<T, tSize>> {
  static constexpr bool is_tag = true;
  static constexpr bool is_vector_tag = true;
  static constexpr bool is_full_vector_tag = false;
};
template<typename TTag>
struct IsTagTrait<GeometryRespectingTag<TTag>> : public IsTagTrait<TTag> {};

template<typename TTag>
concept AnyTag = IsTagTrait<TTag>::is_tag;

template<typename TTag>
concept VectorTag = IsTagTrait<TTag>::is_vector_tag;
template<typename TTag>
concept ScalarTag = AnyTag<TTag> && !VectorTag<TTag>;

template<typename TTag>
concept FullVectorTag = IsTagTrait<TTag>::is_full_vector_tag;
template<typename TTag>
concept PartialVectorTag = VectorTag<TTag> && !FullVectorTag<TTag>;

// tag traits

template<typename TValue, typename TTag>
struct TagTypeTrait;
template<typename TValue>
struct TagTypeTrait<TValue, Scalar> {
  using Type = TValue;
};
template<typename TValue, std::size_t tSize>
struct TagTypeTrait<TValue, VectorSize<tSize>> {
  using Type = Vector<TValue, tSize>;
};
template<typename TValue, typename TTag>
struct TagTypeTrait<TValue, GeometryRespectingTag<TTag>> : public TagTypeTrait<TValue, TTag> {};

template<typename TValue, typename TTag>
using TagType = TagTypeTrait<TValue, TTag>::Type;

template<typename TValue, typename TTag>
struct TagValueTrait;
template<typename TValue, ScalarTag TTag>
struct TagValueTrait<TValue, TTag> {
  using Type = TValue;
};
template<typename TVector, VectorTag TTag>
struct TagValueTrait<TVector, TTag> {
  using Type = TVector::Value;
};
template<typename TValue, typename TTag>
using TagValue = TagValueTrait<TValue, TTag>::Type;

template<typename TTag>
struct IsGeometryRespectingTrait : public std::false_type {};
template<typename TTag>
requires(requires { TTag::is_geometry_respecting; })
struct IsGeometryRespectingTrait<TTag> : public std::bool_constant<TTag::is_geometry_respecting> {};

template<typename TTag>
inline constexpr bool is_geometry_respecting = IsGeometryRespectingTrait<TTag>::value;

// index_from

template<typename TIdx>
inline TIdx index_from(TIdx idx, Scalar /*tag*/) {
  return idx;
}
template<typename TIdx, VectorTag TTag>
inline auto index_from(TIdx idx, TTag /*tag*/) {
  return stado::create_index_from<Vector<TIdx, TTag::size>>(idx);
}

// select

template<typename T>
inline T select(bool selector, T a, T b, ScalarTag auto /*tag*/) {
  return selector ? a : b;
}
template<AnyVector TVec>
inline auto select(auto mask, TVec a, TVec b, VectorTag auto tag) {
  return tag.mask(stado::select(mask, a, b));
}

// load_ptr

template<typename T>
inline T& load_ptr(T* src, Scalar /*tag*/) {
  return *src;
}
template<typename T>
inline T load_ptr(const T* src, Scalar /*tag*/) {
  return *src;
}
template<typename T, std::size_t tSize>
inline Vector<T, tSize> load_ptr(const T* src, VectorSize<tSize> /*tag*/) {
  return stado::create_from_ptr<Vector<T, tSize>>(src);
}
template<typename T, std::size_t tSize>
inline Vector<T, tSize> load_ptr(const T* src, VectorPartSize<tSize> tag) {
  return stado::create_partially_from_ptr<Vector<T, tSize>>(tag.part(), src);
}

// is_load_valid

inline bool is_load_valid(std::size_t remaining, Scalar /*tag*/) {
  return remaining > 0;
}
template<std::size_t tSize>
inline bool is_load_valid(std::size_t remaining, VectorSize<tSize> /*tag*/) {
  return remaining >= tSize;
}
template<std::size_t tSize>
inline bool is_load_valid(std::size_t remaining, VectorPartSize<tSize> tag) {
  return remaining >= tag.part();
}

// load_ptr_extended

inline decltype(auto) load_ptr_extended(auto* src, Scalar tag) {
  return load_ptr(src, tag);
}
template<typename T, std::size_t tSize>
inline Vector<T, tSize> load_ptr_extended(const T* src, VectorSize<tSize> tag) {
  return load_ptr(src, tag);
}
template<typename T, std::size_t tSize>
inline Vector<T, tSize> load_ptr_extended(const T* src, VectorPartSize<tSize> /*tag*/) {
  return load_ptr(src, VectorSize<tSize>{});
}

// store_ptr

template<typename T>
inline void store_ptr(T* dst, T src, Scalar /*tag*/) {
  *dst = src;
}
template<typename T, std::size_t tSize>
inline void store_ptr(T* dst, grex::Vector<T, tSize> src, VectorSize<tSize> /*tag*/) {
  src.store(dst);
}
template<typename T, std::size_t tSize>
inline void store_ptr(T* dst, grex::Vector<T, tSize> src, VectorPartSize<tSize> tag) {
  src.store_partial(tag.part(), dst);
}

// convert_safe

template<typename TDst>
inline auto convert_safe(const auto& src, Scalar /*tag*/) {
  return TDst{src};
}
template<typename TToElement, AnyVector TFrom>
inline auto convert_safe(const TFrom& from, VectorTag auto /*tag*/) {
  using ToVector = Vector<TToElement, TFrom::size>;
  return stado::convert_safe<ToVector>(from);
}

template<typename TToElement, AnyMask TFrom>
inline auto convert_safe(const TFrom& from, VectorTag auto /*tag*/) {
  using ToMask = Mask<TToElement, TFrom::size>;
  return stado::convert_safe<ToMask>(from);
}

// convert_unsafe

template<typename TDst>
inline auto convert_unsafe(const auto& src, Scalar /*tag*/) {
  return static_cast<TDst>(src);
}
template<typename TToElement, AnyVector TFrom>
inline auto convert_unsafe(const TFrom& from, VectorTag auto /*tag*/) {
  using ToVector = grex::Vector<TToElement, TFrom::size>;
  return stado::convert_unsafe<ToVector>(from);
}

// lookup

template<typename T, std::size_t tExtent>
inline T lookup(auto idx, std::span<const T, tExtent> data, Scalar /*tag*/) {
  return data[idx];
}
template<typename T, std::size_t tExtent, std::size_t tSize>
inline Vector<T, tSize> lookup(auto idxs, std::span<const T, tExtent> data,
                               VectorSize<tSize> /*tag*/) {
  auto extended = stado::convert_safe<Vector<grex::FloatSize<T>, tSize>>(idxs);
  return stado::lookup(extended, data.data());
}
template<typename T, std::size_t tExtent, std::size_t tSize>
inline Vector<T, tSize> lookup(auto idxs, std::span<const T, tExtent> data,
                               TypedVectorSize<T, tSize> /*tag*/) {
  return lookup(idxs, data, VectorSize<tSize>{});
}
template<typename T, std::size_t tExtent, std::size_t tSize>
inline Vector<T, tSize> lookup(auto idxs, std::span<const T, tExtent> data,
                               TypedVectorMaskSize<T, tSize> tag) {
  auto extended = stado::convert_safe<Vector<grex::FloatSize<T>, tSize>>(idxs);
  return stado::lookup_masked(tag.mask(), extended, data.data());
}
template<typename T, std::size_t tExtent, std::size_t tSize>
inline Vector<T, tSize> lookup(auto idxs, std::span<const T, tExtent> data,
                               VectorPartSize<tSize> tag) {
  return lookup(idxs, data, tag.instantiate(thes::type_tag<T>));
}

// lookup_masked

template<typename T, std::size_t tExtent>
inline T lookup_masked(bool mask, auto idx, std::span<const T, tExtent> data, Scalar /*tag*/) {
  return mask ? data[idx] : T{};
}
template<typename T, std::size_t tExtent, VectorTag TTag>
inline Vector<T, TTag::size> lookup_masked(AnyMask auto mask, auto idxs,
                                           std::span<const T, tExtent> data, TTag tag) {
  auto extended = stado::convert_safe<Vector<grex::FloatSize<T>, TTag::size>>(idxs);
  return stado::lookup_masked(tag.mask(mask), extended, data.data());
}

// zero

template<Vectorizable T>
inline T zero(Scalar /*tag*/) {
  return T{};
}
template<Vectorizable T>
inline auto zero(VectorTag auto tag) {
  return Vector<T, tag.size>{T{}};
}

// constant

template<typename T>
inline T constant(T value, Scalar /*tag*/) {
  return value;
}
template<typename T, std::size_t tSize>
inline auto constant(T value, VectorSize<tSize> /*tag*/) {
  return Vector<T, tSize>(value);
}
template<typename T, std::size_t tSize>
inline auto constant(T value, TypedVectorSize<T, tSize> /*tag*/) {
  return Vector<T, tSize>(value);
}
template<typename T, std::size_t tSize>
inline auto constant(T value, VectorPartSize<tSize> tag) {
  auto vector = grex::Vector<T, tSize>(value);
  vector.cutoff(tag.part());
  return vector;
}
template<typename T, std::size_t tSize>
inline auto constant(T value, TypedVectorMaskSize<T, tSize> tag) {
  return tag.mask(grex::Vector<T, tSize>(value));
}

// abs

template<Vectorizable T>
inline T abs(T value, Scalar /*tag*/) {
  return std::abs(value);
}
template<AnyVector TVec>
inline TVec abs(TVec value, VectorTag auto /*tag*/) {
  return stado::abs(value);
}

// fma/fms/fnma

template<Vectorizable T>
inline T fma(T x, T y, T z, Scalar /*tag*/) {
  return thes::fast::fma(x, y, z);
}
template<AnyVector TVec>
inline TVec fma(TVec x, TVec y, TVec z, VectorTag auto /*tag*/) {
  return stado::mul_add(x, y, z);
}

template<Vectorizable T>
inline T fms(T x, T y, T z, Scalar /*tag*/) {
  return thes::fast::fma(x, y, -z);
}
template<AnyVector TVec>
inline TVec fms(TVec x, TVec y, TVec z, VectorTag auto /*tag*/) {
  return stado::mul_sub(x, y, z);
}

template<Vectorizable T>
inline T fnma(T x, T y, T z, Scalar /*tag*/) {
  return thes::fast::fma(-x, y, z);
}
template<AnyVector TVec>
inline TVec fnma(TVec x, TVec y, TVec z, VectorTag auto /*tag*/) {
  return stado::nmul_add(x, y, z);
}

// select_add
template<Vectorizable T>
inline T select_add(bool mask, T a, T b, Scalar /*tag*/) {
  return mask ? (a + b) : a;
}
template<AnyVector TVec>
inline TVec select_add(AnyMask auto mask, TVec a, TVec b, VectorTag auto /*tag*/) {
  return stado::if_add(mask, a, b);
}

// select_div
template<Vectorizable T>
inline T select_div(bool mask, T a, T b, Scalar /*tag*/) {
  return mask ? (a / b) : a;
}
template<AnyVector TVec>
inline TVec select_div(AnyMask auto mask, TVec a, TVec b, VectorTag auto /*tag*/) {
  return stado::if_div(mask, a, b);
}

// shuffle_up
template<Vectorizable T>
inline T shuffle_up(T /*x*/, T first, Scalar /*tag*/) {
  return first;
}
template<Vectorizable T, TypeVector<T> TVec>
inline TVec shuffle_up(TVec x, T first, VectorTag auto /*tag*/) {
  return stado::shuffle_up(x, first);
}

// shuffle_down
template<Vectorizable T>
inline T shuffle_down(T /*x*/, T last, Scalar /*tag*/) {
  return last;
}
template<Vectorizable T, TypeVector<T> TVec>
inline TVec shuffle_down(TVec x, T last, VectorTag auto /*tag*/) {
  return stado::shuffle_down(x, last);
}

// is_finite
inline bool is_finite(std::floating_point auto v, Scalar /*tag*/) {
  return std::isfinite(v);
}
inline AnyMask auto is_finite(AnyVector auto vec, VectorTag auto /*tag*/) {
  return stado::is_finite(vec);
}

// max
template<Vectorizable T>
inline T max(T v1, T v2, Scalar /*tag*/) {
  return std::max(v1, v2);
}
template<AnyVector TVec>
inline TVec max(TVec v1, TVec v2, VectorTag auto /*tag*/) {
  return stado::max(v1, v2);
}

// horizontal_add
template<Vectorizable T>
inline T horizontal_add(T value, Scalar /*tag*/) {
  return value;
}
template<AnyVector TVec, std::size_t tSize>
inline TVec::Value horizontal_add(TVec value, VectorSize<tSize> /*tag*/) {
  return stado::horizontal_add(value);
}

// horizontal_max

template<Vectorizable T>
inline T horizontal_max(T value, Scalar /*tag*/) {
  return value;
}
template<AnyVector TVec, std::size_t tSize>
inline TVec::Value horizontal_max(TVec value, VectorSize<tSize> /*tag*/) {
  return stado::horizontal_max(value);
}

// horizontal_and

inline bool horizontal_and(bool mask, Scalar /*tag*/) {
  return mask;
}
template<std::size_t tSize>
inline bool horizontal_and(AnyMask auto mask, VectorSize<tSize> /*tag*/) {
  return stado::horizontal_and(mask);
}
template<AnyMask TMask, std::size_t tSize>
inline bool horizontal_and(TMask mask, VectorPartSize<tSize> tag) {
  return stado::horizontal_and(mask | ~tag.make_mask(thes::type_tag<TMask>));
}
template<typename T, std::size_t tSize>
inline bool horizontal_and(AnyMask auto mask, TypedVectorSize<T, tSize> /*tag*/) {
  return horizontal_and(mask, VectorSize<tSize>{});
}
template<typename T, std::size_t tSize>
inline bool horizontal_and(AnyMask auto mask, TypedVectorMaskSize<T, tSize> tag) {
  return stado::horizontal_and(mask | ~tag.mask());
}
template<AnyTag TTag>
inline bool horizontal_and(AnyMask auto mask, GeometryRespectingTag<TTag> tag) {
  return horizontal_and(mask, static_cast<TTag>(tag));
}

// to_vector

struct ToVectorGenerator {
  template<typename TRange>
  THES_ALWAYS_INLINE constexpr auto operator()(TRange&& range) const {
    using Range = std::decay_t<TRange>;
    using Value = thes::star::Value<Range>;
    constexpr std::size_t size = thes::star::size<Range>;

    return thes::star::to_container<grex::Vector<Value, size>>(std::forward<TRange>(range));
  }
};

inline constexpr ToVectorGenerator to_vector{};
} // namespace grex

namespace thes::star {
template<>
struct ConsumerGeneratorTrait<grex::ToVectorGenerator> : public std::true_type {};
} // namespace thes::star

namespace grex {
// transform

template<typename TSize = std::size_t>
THES_ALWAYS_INLINE inline auto transform(auto op, Scalar /*tag*/) {
  return op(thes::value_tag<TSize, 0>);
}
template<typename TSize = std::size_t, std::size_t tSize>
THES_ALWAYS_INLINE inline auto transform(auto op, VectorSize<tSize> /*tag*/) {
  return thes::star::index_transform<TSize, tSize>([&](auto idx)
                                                     THES_ALWAYS_INLINE { return op(idx); }) |
         to_vector;
}
template<typename TSize = std::size_t, std::size_t tSize>
THES_ALWAYS_INLINE inline auto transform(auto op, VectorPartSize<tSize> tag) {
  return thes::star::index_transform<TSize, tSize>([&](auto idx) THES_ALWAYS_INLINE {
           using Ret = decltype(op(idx));
           return (idx < tag.part()) ? op(idx) : Ret{0};
         }) |
         to_vector;
}

template<typename TSize = std::size_t>
inline void for_each(auto op, thes::TypedValueTag<thes::IterDirection> auto /*iter_tag*/,
                     Scalar /*vec_tag*/) {
  op(thes::value_tag<TSize, 0>);
}
template<typename TSize = std::size_t, std::size_t tSize>
inline void for_each(auto op, thes::TypedValueTag<thes::IterDirection> auto iter_tag,
                     VectorSize<tSize> /*vec_tag*/) {
  if constexpr (iter_tag() == thes::IterDirection::FORWARD) {
    for (TSize i = 0; i < tSize; ++i) {
      op(i);
    }
  } else {
    for (TSize i = tSize; i > 0; --i) {
      op(i - 1);
    }
  }
}
template<typename TSize = std::size_t, std::size_t tSize>
inline auto for_each(auto op, thes::TypedValueTag<thes::IterDirection> auto iter_tag,
                     VectorPartSize<tSize> vec_tag) {
  const auto part = *thes::safe_cast<TSize>(vec_tag.part());
  if constexpr (iter_tag() == thes::IterDirection::FORWARD) {
    for (TSize i = 0; i < part; ++i) {
      op(i);
    }
  } else {
    for (TSize i = part; i > 0; --i) {
      op(i - 1);
    }
  }
}

template<typename TSize = std::size_t>
inline void for_each(auto op, AnyTag auto tag) {
  for_each(std::move(op), thes::auto_tag<thes::IterDirection::FORWARD>, tag);
}
} // namespace grex

#endif // INCLUDE_LINEAL_VECTORIZATION_GREX_HPP
