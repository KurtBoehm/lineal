// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_BASE_TAG_HPP
#define INCLUDE_LINEAL_BASE_TAG_HPP

#include <string_view>
#include <type_traits>

#include "thesauros/macropolis.hpp"
#include "thesauros/types.hpp"

#define LINEAL_DEFINE_BOOL_TAG(NAME, BODY, TRUE_PASCAL, TRUE_SNAKE, FALSE_PASCAL, FALSE_SNAKE) \
  namespace lineal { \
  template<bool tValue> \
  struct NAME##Tag : public thes::BoolTag<tValue> BODY; \
  using TRUE_PASCAL##Tag = NAME##Tag<true>; \
  using FALSE_PASCAL##Tag = NAME##Tag<false>; \
  inline constexpr TRUE_PASCAL##Tag TRUE_SNAKE##_tag{}; \
  inline constexpr FALSE_PASCAL##Tag FALSE_SNAKE##_tag{}; \
  template<typename T> \
  struct Is##NAME##TagTrait : public std::false_type {}; \
  template<bool tValue> \
  struct Is##NAME##TagTrait<NAME##Tag<tValue>> : public std::true_type {}; \
  template<typename T> \
  concept Any##NAME##Tag = Is##NAME##TagTrait<T>::value; \
  template<bool tVal1, bool tVal2> \
  inline constexpr bool operator==(NAME##Tag<tVal1> /*tag1*/, NAME##Tag<tVal2> /*tag2*/) { \
    return tVal1 == tVal2; \
  } \
  } \
  template<> \
  struct thes::SerialValueTrait<lineal::TRUE_PASCAL##Tag> { \
    static constexpr std::string_view make(lineal::TRUE_PASCAL##Tag /*tag*/) { \
      return #TRUE_SNAKE; \
    } \
  }; \
  template<> \
  struct thes::SerialValueTrait<lineal::FALSE_PASCAL##Tag> { \
    static constexpr std::string_view make(lineal::FALSE_PASCAL##Tag /*tag*/) { \
      return #FALSE_SNAKE; \
    } \
  };

#define LINEAL_DEFINE_SIMPLE_BOOL_TAG(NAME, TRUE_PASCAL, TRUE_SNAKE, FALSE_PASCAL, FALSE_SNAKE) \
  LINEAL_DEFINE_BOOL_TAG(NAME, {}, TRUE_PASCAL, TRUE_SNAKE, FALSE_PASCAL, FALSE_SNAKE)

LINEAL_DEFINE_SIMPLE_BOOL_TAG(Direction, Forward, forward, Backward, backward)
LINEAL_DEFINE_SIMPLE_BOOL_TAG(Neighbour, After, after, Before, before)
LINEAL_DEFINE_SIMPLE_BOOL_TAG(Valuation, Valued, valued, Unvalued, unvalued)
LINEAL_DEFINE_SIMPLE_BOOL_TAG(Ordering, Ordered, ordered, Unordered, unordered)
LINEAL_DEFINE_SIMPLE_BOOL_TAG(Exchange, Request, request, Respond, respond)
LINEAL_DEFINE_SIMPLE_BOOL_TAG(ZeroSkipping, SkipZero, skip_zero, UnskipZero, unskip_zero)
LINEAL_DEFINE_SIMPLE_BOOL_TAG(LocalPart, Own, own, Local, local)
LINEAL_DEFINE_SIMPLE_BOOL_TAG(MatrixDimension, MatrixRow, matrix_row, MatrixColumn, matrix_column)
LINEAL_DEFINE_SIMPLE_BOOL_TAG(ZeroHandling, ZeroToOne, zero_to_one, ZeroToZero, zero_to_zero)
LINEAL_DEFINE_SIMPLE_BOOL_TAG(Flushing, Flush, flush, Unflush, unflush)
LINEAL_DEFINE_SIMPLE_BOOL_TAG(Uniqueness, Unique, unique, NonUnique, non_unique)

#undef LINEAL_DEFINE_BOOL_TAG
#undef LINEAL_DEFINE_SIMPLE_BOOL_TAG

#endif // INCLUDE_LINEAL_BASE_TAG_HPP
