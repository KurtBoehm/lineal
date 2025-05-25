// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_LINEAR_SOLVER_ITERATIVE_STATIONARY_JACOBI_SOLVER_HPP
#define INCLUDE_LINEAL_LINEAR_SOLVER_ITERATIVE_STATIONARY_JACOBI_SOLVER_HPP

#include <cassert>
#include <utility>

#include "thesauros/types.hpp"

#include "lineal/base.hpp"
#include "lineal/linear-solver/iterative-stationary/broom.hpp"
#include "lineal/linear-solver/iterative-stationary/jacobi/sweep.hpp"
#include "lineal/parallel/distributed-info/operation.hpp"

namespace lineal {
struct JacobiBroom {
  template<typename TSelf, AnyVector TSolIn, AnyVector TSolOut, AnyVector TRhs>
  static void sweep(TSelf& self, TSolIn&& sol_in, TSolOut& sol_out, const TRhs& rhs,
                    const Env auto& env) {
    using DistributedInfo = TSelf::DistributedInfo;

    using Sweep =
      detail::JacobiSweep<thes::VoidConstLvalRef<DistributedInfo>, typename TSelf::Real,
                          const typename TSelf::Lhs&, const TSolIn&, TSolOut&, const TRhs&>;
    Sweep sweep{
      distributed_info_storage(self.lhs_), self.relax_, self.lhs_,
      std::forward<TSolIn>(sol_in),        sol_out,     rhs,
    };
    env.execution_policy().execute(sweep);
  }
};

template<typename TReal, typename TIterManager>
using JacobiSolver = RelaxedSweepSolver<TReal, SingleSweepImpl<JacobiBroom>, TIterManager>;
} // namespace lineal

#endif // INCLUDE_LINEAL_LINEAR_SOLVER_ITERATIVE_STATIONARY_JACOBI_SOLVER_HPP
