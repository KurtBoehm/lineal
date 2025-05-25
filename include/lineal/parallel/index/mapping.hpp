// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_PARALLEL_INDEX_MAPPING_HPP
#define INCLUDE_LINEAL_PARALLEL_INDEX_MAPPING_HPP

#include <concepts>
#include <limits>

#include "thesauros/utility.hpp"

namespace lineal {
template<std::unsigned_integral TSize>
struct IndexRangeMap {
  using Size = TSize;
  using OptionalSize = thes::ValueOptional<Size, std::numeric_limits<Size>::max()>;

  IndexRangeMap(Size begin, Size end, Size offset) : begin_(begin), end_(end), offset_(offset) {}

  OptionalSize lookup(TSize key) const {
    return ((begin_ <= key) & (key < end_)) ? OptionalSize{key + offset_} : OptionalSize{};
  }

  Size begin() const {
    return begin_;
  }
  Size end() const {
    return end_;
  }
  Size offset() const {
    return offset_;
  }

private:
  TSize begin_;
  TSize end_;
  TSize offset_;
};
} // namespace lineal

#endif // INCLUDE_LINEAL_PARALLEL_INDEX_MAPPING_HPP
