// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_BASE_CONCEPT_VIEW_HPP
#define INCLUDE_LINEAL_BASE_CONCEPT_VIEW_HPP

namespace lineal {
template<typename TTensor>
concept HasLocalView = requires(TTensor tensor) { tensor.local_view(); };
template<typename... TTensors>
concept HaveLocalView = (... && HasLocalView<TTensors>);
} // namespace lineal

#endif // INCLUDE_LINEAL_BASE_CONCEPT_VIEW_HPP
