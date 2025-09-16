// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_TENSOR_DYNAMIC_CSR_CREATE_FROM_HPP
#define INCLUDE_LINEAL_TENSOR_DYNAMIC_CSR_CREATE_FROM_HPP

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <latch>
#include <optional>
#include <span>

#include "thesauros/containers.hpp"
#include "thesauros/static-ranges.hpp"
#include "thesauros/utility.hpp"

#include "lineal/base.hpp"
#include "lineal/parallel.hpp"
#include "lineal/tensor/dynamic/stencil/matrix/shared.hpp"
#include "lineal/tensor/dynamic/stencil/vector/shared.hpp"
#include "lineal/vectorization.hpp"

namespace lineal {
template<AnyMatrix TLhs, AnyVector TRhs, typename TCompressMap>
struct System {
  TLhs lhs;
  TRhs rhs;
  TCompressMap compress_map;
};

// The resulting matrix is compressed if the valuator is uses a zero-to-one mapping
template<SharedMatrix TLhs, SharedVector TRhs, typename TValuator>
static auto base_csr_from_adjacent_stencil(const TValuator& valuator, auto& dist_info_storage,
                                           auto& comm, const auto& env) {
  using Real = TValuator::Real;
  using Size = TValuator::Size;
  using SizeByte = TValuator::SizeByte;
  using LhsValue = TLhs::Value;

  using StencilLhs = AdjacentStencilMatrix<const TValuator&>;
  using StencilRhs = AdjacentStencilVector<const TValuator&>;
  using CompressMap = thes::OptionalMultiByteIntegers<SizeByte, grex::register_bytes.back()>;
  using ExtRowIdx = StencilLhs::ExtRowIdx;
  using ColIdx = StencilLhs::ColumnIdx;
  static_assert(std::same_as<ExtRowIdx, ColIdx>);

  constexpr bool is_shared = TValuator::is_shared;
  constexpr IsSymmetric is_symmetric{TValuator::is_symmetric};
  constexpr bool compress = !TValuator::zero_to_one;
  constexpr bool dist_comp = !is_shared && compress;

  decltype(auto) base_dist_info = distributed_info_storage(valuator);
  const auto& expo = env.execution_policy();

  StencilLhs stencil_lhs{valuator};
  StencilRhs stencil_rhs{valuator};

  auto compress_map = [&] {
    if constexpr (compress) {
      return CompressMap::create_empty(stencil_lhs.ext_row_num());
    } else {
      return thes::Empty{};
    }
  }();
  const auto& cmap = compress_map;

  const auto tnum = expo.thread_num();
  const auto tnum_ptrdiff = *thes::safe_cast<std::ptrdiff_t>(tnum);
  auto lhs_planner = TLhs::multithread_planner(unique_tag, local_index_tag);
  lhs_planner.initialize(tnum, is_symmetric);
  auto lhs_builder = TLhs::multithread_builder(thes::type_tag<Real>, unique_tag, local_index_tag);
  std::optional<TRhs> rhs_opt{};
  auto barriers =
    thes::star::generate<(dist_comp ? 4 : 3)>([tnum_ptrdiff] { return std::latch{tnum_ptrdiff}; }) |
    thes::star::to_array;

  expo.execute_segmented(
    stencil_lhs.ext_row_num(),
    [&](auto thread_idx, Size idx0, Size idx1) {
      const ExtRowIdx eidx0{idx0};
      const ExtRowIdx eidx1{idx1};

      {
        auto thread_instance = lhs_planner.thread_instance(thread_idx);
        for (decltype(auto) row :
             thes::value_range(stencil_lhs.ext_iter_at(eidx0), stencil_lhs.ext_iter_at(eidx1))) {
          auto diag = [&](ColIdx i, Real v) {
            if constexpr (compress) {
              if (v != 0) {
                ExtRowIdx row_idx = thread_instance.row_index();
                compress_map[index_value(i)] = index_value(row_idx);
                thread_instance.add_column(i);
                ++thread_instance;
              }
            } else {
              assert(v > 0);
              thread_instance.add_column(i);
              ++thread_instance;
            }
          };
          auto off_diag = [&](ColIdx j, Real v) {
            if (v != 0) {
              thread_instance.add_column(j);
            }
          };
          row.iterate(off_diag, diag, off_diag, valued_tag, unordered_tag);
        }

        thread_instance.finalize();
      }

      std::get<0>(barriers).arrive_and_wait();

      const ExtRowIdx row_offset = lhs_planner.row_offset(thread_idx);
      const auto non_zero_offset = lhs_planner.non_zero_offset(thread_idx);

      std::get<1>(barriers).arrive_and_wait();

      if (thread_idx == 0) {
        if constexpr (!dist_comp) {
          lhs_builder.initialize(std::move(lhs_planner), base_dist_info);
          if constexpr (requires { rhs_opt.emplace(lhs_builder.row_num(), base_dist_info); }) {
            rhs_opt.emplace(lhs_builder.row_num(), base_dist_info);
          } else {
            rhs_opt.emplace(lhs_builder.row_num());
          }
        }
      } else if constexpr (compress) {
        for (const auto index : thes::range(idx0, idx1)) {
          decltype(auto) cidx = compress_map[index];
          typename CompressMap::Value cidxv = cidx;
          if (cidxv.has_value()) {
            cidx = *cidxv + index_value(row_offset);
          }
        }
      }

      std::get<2>(barriers).arrive_and_wait();

      [[maybe_unused]] const auto& dist_info = [&]() -> const auto& {
        if constexpr (dist_comp) {
          if (thread_idx == 0) {
            const Size own_begin = base_dist_info.own_begin(local_index_tag);
            const Size own_end = base_dist_info.own_end(local_index_tag);
            const Size rown_begin = base_dist_info.size(after_tag);
            const Size rown_end = rown_begin + base_dist_info.size(own_index_tag);

            auto min_it = std::find_if(cmap.begin() + own_begin, cmap.begin() + own_end,
                                       [](auto v) { return v.has_value(); });
            assert(min_it != cmap.begin() + own_end);
            auto max_it = std::find_if(cmap.rbegin() + rown_begin, cmap.rbegin() + rown_end,
                                       [](auto v) { return v.has_value(); });
            assert(max_it != cmap.rbegin() + rown_end);

            const Size comp_own_begin = min_it->value();
            const Size comp_own_end = max_it->value() + 1;
            const Size comp_local_end = lhs_planner.row_num();

            thes::DynamicArray<Size> own_sizes(*thes::safe_cast<std::size_t>(comm.size()));
            comm.allgather(comp_own_end - comp_own_begin, std::span{own_sizes});
            const Size global_own_off =
              std::reduce(own_sizes.begin(), own_sizes.begin() + comm.rank(), Size{0});
            const Size global_local_off = global_own_off - comp_own_begin;
            const Size global_size = std::reduce(own_sizes.begin(), own_sizes.end(), Size{0});

            dist_info_storage.emplace(global_local_off, comp_own_begin + global_local_off,
                                      comp_own_end + global_local_off,
                                      comp_local_end + global_local_off, global_size, comm);
            assert(dist_info_storage.has_value());
            const auto& dinfo = *dist_info_storage;

            lhs_builder.initialize(std::move(lhs_planner), dinfo);
            if constexpr (requires { rhs_opt.emplace(lhs_builder.row_num(), dinfo); }) {
              rhs_opt.emplace(lhs_builder.row_num(), dinfo);
            } else {
              rhs_opt.emplace(lhs_builder.row_num());
            }
          }
          std::get<3>(barriers).arrive_and_wait();

          assert(dist_info_storage.has_value());
          return *dist_info_storage;
        } else {
          return base_dist_info;
        }
      }();

      auto thread_instance = lhs_builder.thread_instance(row_offset, non_zero_offset);
      auto vector_it = rhs_opt.value().iter_at(row_offset);
      auto mat_it = stencil_lhs.ext_iter_at(eidx0);
      auto vec_it = stencil_rhs.ext_iter_at(eidx0);
      for (const Size i : thes::range(idx0, idx1)) {
        [[maybe_unused]] ExtRowIdx idx{i};
        ExtRowIdx idx_com{i};
        if constexpr (compress) {
          const auto coidx = cmap[i];
          if (!coidx.has_value()) {
            ++mat_it, ++vec_it;
            continue;
          }
          idx_com = ExtRowIdx{*coidx};
        }

        auto off_diag = [&](ColIdx j, Real v) {
          if (v != 0) {
            auto jcom = [&] {
              if constexpr (compress) {
                return ColIdx{cmap[index_value(j)].value()};
              } else {
                return j;
              }
            }();
            thread_instance.insert(jcom, static_cast<LhsValue>(v));
          }
        };
        auto diag = [&]([[maybe_unused]] ColIdx j, Real v) {
          assert(j == index_convert<ColIdx>(idx, dist_info) && v != 0);
          thread_instance.insert(idx_com, static_cast<LhsValue>(v));
          *vector_it++ = *vec_it;
          ++thread_instance;
        };
        mat_it->iterate(off_diag, diag, off_diag, valued_tag, unordered_tag);
        ++mat_it, ++vec_it;
      }
      thread_instance.finalize();
    },
    tnum);

  return System{
    .lhs = std::move(lhs_builder).build(),
    .rhs = std::move(rhs_opt).value(),
    .compress_map = std::move(compress_map),
  };
}

template<SharedMatrix TLhs, SharedVector TRhs, typename TValuator>
static auto csr_from_adjacent_stencil(const TValuator& valuator, const auto& env) {
  thes::Empty e0{};
  thes::Empty e1{};
  return base_csr_from_adjacent_stencil<TLhs, TRhs>(valuator, e0, e1, env);
}
} // namespace lineal

#endif // INCLUDE_LINEAL_TENSOR_DYNAMIC_CSR_CREATE_FROM_HPP
