// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_COMPONENT_WISE_FACADE_COMBINED_HPP
#define INCLUDE_LINEAL_COMPONENT_WISE_FACADE_COMBINED_HPP

#include "lineal/base.hpp"
#include "lineal/component-wise/facade/shared.hpp"

namespace lineal::facades {
template<typename TDerived, typename TProvider, typename... TChildren>
struct ComponentWiseOpTrait;
template<typename TDerived, typename TProvider, SharedRange... TChildren>
struct ComponentWiseOpTrait<TDerived, TProvider, TChildren...> {
  using Type = SharedComponentWiseOp<TDerived, TProvider, TChildren...>;
};
template<typename TDerived, typename TProvider, SharedRanges... TChildren>
requires(sizeof...(TChildren) > 0)
using ComponentWiseOp = ComponentWiseOpTrait<TDerived, TProvider, TChildren...>::Type;
} // namespace lineal::facades

#endif // INCLUDE_LINEAL_COMPONENT_WISE_FACADE_COMBINED_HPP
