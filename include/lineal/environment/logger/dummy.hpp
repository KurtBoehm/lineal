// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_ENVIRONMENT_LOGGER_DUMMY_HPP
#define INCLUDE_LINEAL_ENVIRONMENT_LOGGER_DUMMY_HPP

#include <string_view>

namespace lineal {
struct DummyLogger {
  [[nodiscard]] DummyLogger add_array(const auto& /*name*/) const {
    return {};
  }
  [[nodiscard]] DummyLogger add_object(const auto& /*name*/) const {
    return {};
  }
  [[nodiscard]] DummyLogger add_object() const {
    return {};
  }
  [[nodiscard]] DummyLogger add_dummy() const {
    return {};
  }

  void log(std::string_view /*key*/, const auto& /*value*/) const {}
  void log(const auto& /*value*/) const {}
  void log_named(const auto& /*value*/) const {}
};
} // namespace lineal

#endif // INCLUDE_LINEAL_ENVIRONMENT_LOGGER_DUMMY_HPP
