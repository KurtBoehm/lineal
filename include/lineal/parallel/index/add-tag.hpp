// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_PARALLEL_INDEX_ADD_TAG_HPP
#define INCLUDE_LINEAL_PARALLEL_INDEX_ADD_TAG_HPP

#include "lineal/parallel/def/index.hpp"

namespace lineal {
template<typename TInType, OptIndexTag TIdxTag>
struct TagAdder {
  using Type = TInType;

  static constexpr Type convert(Type idx) {
    return idx;
  }
};

template<OptIndexTag TIdxTag, typename TInType>
inline constexpr auto add_tag(TInType input) {
  return TagAdder<TInType, TIdxTag>::convert(input);
}
} // namespace lineal

#endif // INCLUDE_LINEAL_PARALLEL_INDEX_ADD_TAG_HPP
