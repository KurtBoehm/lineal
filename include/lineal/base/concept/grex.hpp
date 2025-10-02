// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_BASE_CONCEPT_GREX_HPP
#define INCLUDE_LINEAL_BASE_CONCEPT_GREX_HPP

#include "grex/base.hpp"
#include "grex/tags.hpp"

namespace lineal {
template<typename TTag, typename TValue>
concept TensorTag = (grex::Vectorizable<TValue> && grex::AnyTag<TTag>) || grex::AnyScalarTag<TTag>;
template<typename TTag, typename TValue>
concept TensorVectorTag =
  (grex::Vectorizable<TValue> && grex::AnyVectorTag<TTag>) || grex::AnyScalarTag<TTag>;
} // namespace lineal

#endif // INCLUDE_LINEAL_BASE_CONCEPT_GREX_HPP
