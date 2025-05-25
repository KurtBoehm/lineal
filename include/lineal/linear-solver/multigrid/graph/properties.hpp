// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_GRAPH_PROPERTIES_HPP
#define INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_GRAPH_PROPERTIES_HPP

#include "thesauros/types/primitives.hpp"

namespace lineal::amg {
struct EdgeProperties {
  static constexpr thes::u8 depends_bit = 0;
  static constexpr thes::u8 depends_mask = 1U << depends_bit;
  static constexpr thes::u8 influences_bit = 1;
  static constexpr thes::u8 influences_mask = 1U << influences_bit;

  EdgeProperties(bool depends, bool influences) : depends_(depends), influences_(influences) {}

  [[nodiscard]] bool depends() const {
    return depends_;
  }
  [[nodiscard]] bool influences() const {
    return influences_;
  }

  void set(bool depends, bool influences) {
    depends_ = depends;
    influences_ = influences;
  }

  [[nodiscard]] bool is_one_way() const {
    return !influences_ && depends_;
  }
  [[nodiscard]] bool is_two_way() const {
    return influences_ && depends_;
  }
  [[nodiscard]] bool is_strong() const {
    return influences_ || depends_;
  }

  [[nodiscard]] static EdgeProperties from_byte(thes::u8 byte) {
    return {(byte & depends_mask) != 0, (byte & influences_mask) != 0};
  }
  [[nodiscard]] thes::u8 to_byte() const {
    return thes::u8(thes::u8{depends_} << depends_bit) |
           thes::u8(thes::u8{influences_} << influences_bit);
  }

private:
  // Whether the tail of the edge depends on its head.
  bool depends_ : 1;
  // Whether the head of the edge is influenced by its tail.
  bool influences_ : 1;
};
} // namespace lineal::amg

#endif // INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_GRAPH_PROPERTIES_HPP
