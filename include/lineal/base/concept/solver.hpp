// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_BASE_CONCEPT_SOLVER_HPP
#define INCLUDE_LINEAL_BASE_CONCEPT_SOLVER_HPP

#include <type_traits>

namespace lineal {
#ifdef DEFINE_SOLVER
#error "DEFINE_SOLVER is already defined!"
#endif
#define DEFINE_SOLVER(NAME) \
  struct Shared##NAME##SolverBase {}; \
  template<typename T> \
  struct IsShared##NAME##SolverTrait : public std::is_base_of<Shared##NAME##SolverBase, T> {}; \
  template<typename T> \
  concept Shared##NAME##Solver = IsShared##NAME##SolverTrait<std::decay_t<T>>::value; \
\
  struct Distributed##NAME##SolverBase {}; \
  template<typename T> \
  struct IsDistributed##NAME##SolverTrait \
      : public std::is_base_of<Distributed##NAME##SolverBase, T> {}; \
  template<typename T> \
  concept Distributed##NAME##Solver = IsDistributed##NAME##SolverTrait<std::decay_t<T>>::value; \
\
  template<typename T> \
  concept NAME##Solver = Shared##NAME##Solver<T> || Distributed##NAME##Solver<T>; \
  template<typename T> \
  concept Opt##NAME##Solver = NAME##Solver<T> || std::is_void_v<T>;

DEFINE_SOLVER(NonStationaryIterative)
DEFINE_SOLVER(StationaryIterative)
DEFINE_SOLVER(Direct)

#undef DEFINE_SOLVER
} // namespace lineal

#endif // INCLUDE_LINEAL_BASE_CONCEPT_SOLVER_HPP
