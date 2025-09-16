// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_COMPONENT_WISE_EXEC_CONSTRAINT_HPP
#define INCLUDE_LINEAL_COMPONENT_WISE_EXEC_CONSTRAINT_HPP

#include <concepts>
#include <type_traits>

#include "thesauros/static-ranges.hpp"
#include "thesauros/utility.hpp"

#include "lineal/base.hpp"

namespace lineal {
template<AnyDirectionTag auto tDir>
struct IterationDirectionConstraint {
  static constexpr auto direction = tDir;
};
using ForwardIterConstraint = IterationDirectionConstraint<forward_tag>;
using BackwardIterConstraint = IterationDirectionConstraint<backward_tag>;

template<AnyDirectionTag auto tDir1, AnyDirectionTag auto tDir2>
constexpr bool operator==(IterationDirectionConstraint<tDir1> /*con1*/,
                          IterationDirectionConstraint<tDir2> /*con2*/) {
  return tDir1 == tDir2;
}

struct MatrixNnzPatternBase {};
struct AdjacentStencilNnzPattern : public MatrixNnzPatternBase {};
struct ArbitraryNnzPattern : public MatrixNnzPatternBase {};
template<typename T>
concept AnyMatrixNnzPatternTag = std::derived_from<T, MatrixNnzPatternBase>;

template<typename T>
inline constexpr auto nnz_pattern = [] {
  if constexpr (requires { T::nnz_pattern; }) {
    return T::nnz_pattern;
  } else {
    return ArbitraryNnzPattern{};
  }
}();

// Either ascending or descending iteration order on each thread
// (which can be specified using the iteration) on contiguous index set segments
struct ThreadSeqIterConstraint {};
// The iteration order ensures that, within each thread, a given index i is visited
// after all indices j < i corresponding to (potentially) non-zero columns in row i
// (according to the given pattern) have been visited.
// Any unknown pattern should be treated like `ArbitraryNnzPattern`.
template<AnyMatrixNnzPatternTag auto tPattern>
struct ThreadKeepNbOrderConstraint {
  static constexpr auto nnz_pattern = tPattern;
};

template<typename T>
inline constexpr auto exec_constraints = [] {
  if constexpr (requires { T::exec_constraints; }) {
    return T::exec_constraints;
  } else {
    return thes::Tuple{};
  }
}();

// When using SIMD vectorization, the vector tag only includes indices which, when converted
// to positions according to the geometry, differ only in the last dimension.
// In row-major 2D, this means that the selected indices in a SIMD vector never cross rows.
struct GeometryPreservingVectorizationConstraint {};

template<typename TChildren>
inline constexpr auto merged_exec_constraints =
  thes::star::index_transform<thes::star::size<TChildren>>(
    [](auto idx) { return exec_constraints<std::decay_t<thes::star::Element<idx, TChildren>>>; }) |
  thes::star::join | thes::star::to_tuple;
} // namespace lineal

#endif // INCLUDE_LINEAL_COMPONENT_WISE_EXEC_CONSTRAINT_HPP
