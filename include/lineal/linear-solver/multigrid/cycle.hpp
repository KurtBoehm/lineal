// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_CYCLE_HPP
#define INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_CYCLE_HPP

#include <cassert>
#include <cstddef>
#include <stdexcept>

#include "thesauros/types.hpp"

namespace lineal {
enum struct CycleKind : thes::u8 {
  // Examples: V and W cycle
  REGULAR,
  // Examples: F cycle
  DECREASING,
};

struct DefaultCycleInfo {
  DefaultCycleInfo(CycleKind cycle, std::size_t iter_num)
      : cycle_(cycle), iter_(iter_num), iter_num_(iter_num) {}

  DefaultCycleInfo& operator++() {
    assert(iter_ > 0);
    --iter_;
    return *this;
  }

  [[nodiscard]] bool is_begin() const {
    return iter_ == iter_num_;
  }

  [[nodiscard]] bool is_end() const {
    return iter_ == 0;
  }

  [[nodiscard]] DefaultCycleInfo get_coarser() const {
    switch (cycle_) {
      case CycleKind::REGULAR: return {cycle_, iter_num_};
      case CycleKind::DECREASING: return {cycle_, iter_};
    }
    throw std::runtime_error("Unknown cycle kind!");
  }

private:
  CycleKind cycle_;
  std::size_t iter_;
  std::size_t iter_num_;
};

struct DefaultCycle {
  using CycleInfo = DefaultCycleInfo;

  DefaultCycle(CycleKind cycle, std::size_t cycle_iter_num)
      : cycle_(cycle), cycle_iter_num_(cycle_iter_num) {}

  [[nodiscard]] CycleInfo fine_cycle_info() const {
    return {cycle_, cycle_iter_num_};
  }
  [[nodiscard]] CycleInfo get_coarsest() const {
    return {cycle_, 0};
  }

private:
  CycleKind cycle_;
  std::size_t cycle_iter_num_;
};
} // namespace lineal

#endif // INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_CYCLE_HPP
