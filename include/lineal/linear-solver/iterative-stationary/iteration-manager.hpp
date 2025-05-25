// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_LINEAR_SOLVER_ITERATIVE_STATIONARY_ITERATION_MANAGER_HPP
#define INCLUDE_LINEAL_LINEAR_SOLVER_ITERATIVE_STATIONARY_ITERATION_MANAGER_HPP

#include <cstddef>

namespace lineal {
struct FixedIterationManager {
  using Size = std::size_t;

  explicit FixedIterationManager(Size iter_num) : iter_num_(iter_num) {}

  [[nodiscard]] constexpr Size iteration_num() const {
    return iter_num_;
  }

private:
  Size iter_num_;
};
} // namespace lineal

#endif // INCLUDE_LINEAL_LINEAR_SOLVER_ITERATIVE_STATIONARY_ITERATION_MANAGER_HPP
