// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_LINEAR_SOLVER_ITERATIVE_STATIONARY_GAUSS_SEIDEL_VARIANT_SOLVER_HPP
#define INCLUDE_LINEAL_LINEAR_SOLVER_ITERATIVE_STATIONARY_GAUSS_SEIDEL_VARIANT_SOLVER_HPP

#include <cassert>
#include <utility>

#include "thesauros/types.hpp"

#include "lineal/base.hpp"
#include "lineal/linear-solver/iterative-stationary/broom.hpp"
#include "lineal/linear-solver/iterative-stationary/gauss-seidel-variant/sweep.hpp"
#include "lineal/parallel/distributed-info/operation.hpp"

namespace lineal {
template<AnyDirectionTag auto tDir, SorVariant tVariant>
struct SorBroom {
  template<typename TSelf, AnyVector TSolIn, AnyVector TSolOut, AnyVector TRhs>
  static void sweep(TSelf& self, TSolIn&& sol_in, TSolOut& sol_out, const TRhs& rhs,
                    const Env auto& env) {
    using DistributedInfo = TSelf::DistributedInfo;

    using Sweep = detail::SorSweep<thes::VoidConstLvalRef<DistributedInfo>, tVariant, tDir,
                                   typename TSelf::Real, const typename TSelf::Lhs&, const TSolIn&,
                                   TSolOut&, const TRhs&>;
    Sweep sweep{
      distributed_info_storage(self.lhs_),
      self.relax_,
      self.lhs_,
      std::forward<TSolIn>(sol_in),
      sol_out,
      rhs,
      self.min_per_thread_,
    };
    env.execution_policy().execute(sweep);
  }
};

template<SorVariant tVariant>
struct SsorBroom {
  template<typename TSelf, AnyVector TSolIn, AnyVector TSolOut, AnyVector TRhs, typename TAux>
  static void sweep(TSelf& self, TSolIn&& sol_in, TSolOut& sol_out, const TRhs& rhs, TAux&& aux,
                    const Env auto& env)
  requires(thes::star::size<TAux> >= 1)
  {
    using DistributedInfo = TSelf::DistributedInfo;
    using Real = TSelf::Real;
    using Lhs = TSelf::Lhs;

    AnyVector decltype(auto) aux_vec = thes::star::get_at<0>(std::forward<TAux>(aux));
    using AuxVec = decltype(aux_vec);
    const auto& expo = env.execution_policy();

    {
      using Sweep = detail::SorSweep<thes::VoidConstLvalRef<DistributedInfo>, tVariant, forward_tag,
                                     Real, const Lhs&, const TSolIn&, AuxVec&, const TRhs&>;
      Sweep sweep{
        distributed_info_storage(self.lhs_),
        self.relax_,
        self.lhs_,
        std::forward<TSolIn>(sol_in),
        aux_vec,
        rhs,
        self.min_per_thread_,
      };
      expo.execute(sweep);
    }
    {
      using Sweep =
        detail::SorSweep<thes::VoidConstLvalRef<DistributedInfo>, tVariant, backward_tag, Real,
                         const Lhs&, AuxVec&, TSolOut&, const TRhs&>;
      Sweep sweep{
        distributed_info_storage(self.lhs_),
        self.relax_,
        self.lhs_,
        aux_vec,
        sol_out,
        rhs,
        self.min_per_thread_,
      };
      expo.execute(sweep);
    }
  }
};

template<typename TReal, typename TIterManager, AnyDirectionTag auto tDir, SorVariant tVariant>
using SorSolver =
  RelaxedSweepSolver<TReal, SingleSweepImpl<SorBroom<tDir, tVariant>>, TIterManager>;

template<typename TReal, typename TIterManager, SorVariant tVariant>
using SsorSolver = RelaxedSweepSolver<TReal, DoubleSweepImpl<SsorBroom<tVariant>>, TIterManager>;
} // namespace lineal

#endif // INCLUDE_LINEAL_LINEAR_SOLVER_ITERATIVE_STATIONARY_GAUSS_SEIDEL_VARIANT_SOLVER_HPP
