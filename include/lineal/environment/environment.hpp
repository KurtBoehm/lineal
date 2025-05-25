// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_ENVIRONMENT_ENVIRONMENT_HPP
#define INCLUDE_LINEAL_ENVIRONMENT_ENVIRONMENT_HPP

#include <type_traits>
#include <utility>

#include "lineal/base/concept.hpp"
#include "lineal/environment/logger/dummy.hpp"

namespace lineal {
template<typename TExecPolicy, typename TLogger>
struct Environment : public EnvBase {
  using ExecutionPolicy = std::decay_t<TExecPolicy>;
  using Logger = std::decay_t<TLogger>;

  Environment(TExecPolicy&& execution_policy, TLogger&& logger)
      : expo_(std::forward<TExecPolicy>(execution_policy)), logger_(std::forward<TLogger>(logger)) {
  }
  Environment(std::in_place_t /*tag*/, auto make_executor, TLogger&& logger)
      : expo_(make_executor()), logger_(std::forward<TLogger>(logger)) {}

  const ExecutionPolicy& execution_policy() const {
    return expo_;
  }
  const Logger& logger() const {
    return logger_;
  }

  auto add_object() const {
    return ::lineal::Environment{expo_, logger_.add_object()};
  }
  auto add_object(const auto& name) const {
    return ::lineal::Environment(expo_, logger_.add_object(name));
  }
  decltype(auto) add_object(const auto& name, auto op) const {
    return op(add_object(name));
  }

  auto add_array(const auto& name) const {
    return ::lineal::Environment(expo_, logger_.add_array(name));
  }
  decltype(auto) add_array(const auto& name, auto op) const {
    return op(add_array(name));
  }

  auto dummy() const {
    return ::lineal::Environment(expo_, DummyLogger{});
  }

  void log(const auto& value) const {
    logger_.log(value);
  }
  void log(const auto& key, const auto& value) const {
    logger_.log(key, value);
  }
  void log_named(const auto& value) const {
    logger_.log_named(value);
  }

private:
  TExecPolicy expo_;
  TLogger logger_;
};
template<typename TMkExec, typename TLogger>
Environment(std::in_place_t, TMkExec, TLogger&&)
  -> Environment<std::invoke_result_t<TMkExec>, TLogger>;

template<typename TExecPolicy, typename TLogger>
Environment(TExecPolicy&& execution_policy, TLogger&& logger) -> Environment<TExecPolicy, TLogger>;
} // namespace lineal

#endif // INCLUDE_LINEAL_ENVIRONMENT_ENVIRONMENT_HPP
