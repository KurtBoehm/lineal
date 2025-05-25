// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef TEST_AUX_MSG_HPP
#define TEST_AUX_MSG_HPP

#include "thesauros/macropolis.hpp"

namespace lineal::msg {
template<typename TReal>
struct RefValue {
  THES_DEFINE_TYPE(SNAKE_CASE(RefValue), CONSTEXPR_CONSTRUCTOR,
                   MEMBERS((KEEP(value), TReal), (KEEP(reference), TReal)))
};

template<typename TReal>
struct ResidualErrorBound {
  THES_DEFINE_TYPE(SNAKE_CASE(ResidualErrorBound), CONSTEXPR_CONSTRUCTOR,
                   MEMBERS((KEEP(residual_l2), TReal), (KEEP(error_max), TReal),
                           (KEEP(bound), TReal)))
};
template<typename TReal>
struct ResidualsBound {
  THES_DEFINE_TYPE(SNAKE_CASE(ResidualsBound), CONSTEXPR_CONSTRUCTOR,
                   MEMBERS((KEEP(residual_l2), TReal), (KEEP(residual_max), TReal),
                           (KEEP(bound), TReal)))
};
} // namespace lineal::msg

#endif // TEST_AUX_MSG_HPP
