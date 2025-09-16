// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_PARALLEL_DEF_INDEX_HPP
#define INCLUDE_LINEAL_PARALLEL_DEF_INDEX_HPP

#include <concepts>
#include <type_traits>

#include "thesauros/types/primitives.hpp"

namespace lineal {
enum struct DistributedIndexKind : thes::u8 { OWN, LOCAL, GLOBAL };

template<DistributedIndexKind tIdxKind>
struct DistributedIndexKindTag {
  static constexpr DistributedIndexKind index_kind = tIdxKind;
};

template<DistributedIndexKind tKind1, DistributedIndexKind tKind2>
constexpr bool operator==(DistributedIndexKindTag<tKind1> /*tag1*/,
                          DistributedIndexKindTag<tKind2> /*tag2*/) {
  return tKind1 == tKind2;
}

using GlobalIndexTag = DistributedIndexKindTag<DistributedIndexKind::GLOBAL>;
using LocalIndexTag = DistributedIndexKindTag<DistributedIndexKind::LOCAL>;
using OwnIndexTag = DistributedIndexKindTag<DistributedIndexKind::OWN>;

template<bool tIsShared>
using OptGlobalIndexTag = std::conditional_t<tIsShared, void, GlobalIndexTag>;
template<bool tIsShared>
using OptLocalIndexTag = std::conditional_t<tIsShared, void, LocalIndexTag>;
template<bool tIsShared>
using OptOwnIndexTag = std::conditional_t<tIsShared, void, OwnIndexTag>;

inline constexpr GlobalIndexTag global_index_tag{};
inline constexpr LocalIndexTag local_index_tag{};
inline constexpr OwnIndexTag own_index_tag{};

template<typename T>
concept AnyIndexTag =
  std::same_as<T, GlobalIndexTag> || std::same_as<T, LocalIndexTag> || std::same_as<T, OwnIndexTag>;
template<typename T>
concept OptIndexTag = AnyIndexTag<T> || std::is_void_v<T>;
} // namespace lineal

#endif // INCLUDE_LINEAL_PARALLEL_DEF_INDEX_HPP
