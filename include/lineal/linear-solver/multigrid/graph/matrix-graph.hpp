// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_GRAPH_MATRIX_GRAPH_HPP
#define INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_GRAPH_MATRIX_GRAPH_HPP

#include <cassert>
#include <compare>
#include <cstddef>
#include <functional>
#include <optional>
#include <type_traits>
#include <utility>

#include "thesauros/algorithms.hpp"
#include "thesauros/containers.hpp"
#include "thesauros/iterator.hpp"
#include "thesauros/macropolis.hpp"
#include "thesauros/memory.hpp"
#include "thesauros/types.hpp"

#include "lineal/base.hpp"
#include "lineal/linear-solver/multigrid/types.hpp"
#include "lineal/parallel.hpp"
#include "lineal/vectorization.hpp"

namespace lineal::amg {
template<SharedMatrix TMat, typename TTransform,
         typename TByteAlloc = thes::HugePagesAllocator<std::byte>>
struct MatrixGraph : public SharedGraphBase {
  using Transform = std::decay_t<TTransform>;
  using Matrix = std::decay_t<TMat>;
  using DistributedInfo = DistributedInfoOf<Matrix>;
  using DistributedInfoStorage = thes::VoidStorage<DistributedInfo>;
  static constexpr bool is_shared = std::is_void_v<DistributedInfo>;
  static constexpr auto exec_constraints = lineal::exec_constraints<Matrix>;

  using Weight = Matrix::Value;

  using VertexSizeByte = Matrix::SizeByte;
  using VertexSize = Matrix::Size;
  using VertexIdx = OptOwnIndex<is_shared, VertexSize>;
  using ExtVertexIdx = OptLocalIndex<is_shared, VertexSize>;
  using Vertex = amg::Vertex<VertexIdx>;
  using ExtVertex = amg::Vertex<ExtVertexIdx>;
  using Size = VertexSize;

  using EdgeSizeByte = Matrix::NonZeroSizeByte;
  using EdgeSize = Matrix::NonZeroSize;
  using Edge = amg::Edge<VertexIdx, EdgeSize>;
  using ExtEdge = amg::Edge<ExtVertexIdx, EdgeSize>;

  using EdgeIdxs = thes::MultiByteIntegers<EdgeSizeByte, grex::max_vector_bytes, TByteAlloc>;
  using EdgeIdxsIter = EdgeIdxs::const_iterator;

  using Row = Matrix::ConstRow;
  using RowIter = Matrix::const_iterator;

  struct FullVertex {
    using DistInfo = thes::VoidConstLvalRef<DistributedInfo>;
    using DistInfoStorage = thes::VoidStorage<DistInfo>;

    explicit FullVertex(Row&& row, EdgeSize start_idx)
        : row_(std::forward<Row>(row)), start_idx_(start_idx) {}

    VertexIdx index() const {
      return row_.index();
    }
    Vertex vertex() const {
      return Vertex(row_.index());
    }

    VertexSize full_adjacent_num() const {
      return row_.offdiagonal_num();
    }

    const Row& row() const {
      return row_;
    }
    Weight weight() const {
      return Transform::diagonal(row_.diagonal(), grex::Scalar{});
    }

    EdgeSize start_index() const {
      return start_idx_;
    }

    THES_ALWAYS_INLINE decltype(auto) iterate(auto vtx_op, auto edge_op,
                                              AnyValuationTag auto is_valued,
                                              AnyOrderingTag auto is_ordered) const {
      return iterate_impl(vtx_op, edge_op, is_valued, is_ordered, thes::false_tag);
    }
    THES_ALWAYS_INLINE decltype(auto) iterate_ext(auto vtx_op, auto edge_op,
                                                  AnyValuationTag auto is_valued,
                                                  AnyOrderingTag auto is_ordered) const {
      return iterate_impl(vtx_op, edge_op, is_valued, is_ordered, thes::true_tag);
    }

  private:
    THES_ALWAYS_INLINE
    decltype(auto) iterate_impl(auto vtx_op, auto edge_op, ValuedTag is_valued,
                                AnyOrderingTag auto is_ordered,
                                thes::AnyBoolTag auto is_extended) const {
      decltype(auto) dist_info = row_.distributed_info_storage();
      const VertexIdx vidx = row_.index();
      const ExtVertexIdx exvidx = row_.ext_index();
      EdgeSize edge_idx = start_idx_;

      if constexpr (is_shared) {
        auto offdiag_op = [&](ExtVertexIdx j, Weight value) THES_ALWAYS_INLINE {
          return edge_op(Edge{edge_idx++, vidx, j}, Transform::offdiagonal(value, grex::Scalar{}));
        };
        return row_.iterate(
          offdiag_op,
          [&](ExtVertexIdx j, auto value) THES_ALWAYS_INLINE {
            return vtx_op(Vertex{j}, Transform::diagonal(value, grex::Scalar{}));
          },
          offdiag_op, is_valued, is_ordered);
      } else {
        decltype(auto) range = dist_info.index_range_within(own_index_tag, local_index_tag);
        auto offdiag_op = [&](ExtVertexIdx j, Weight value) THES_ALWAYS_INLINE {
          const auto eidx = edge_idx++;
          if constexpr (is_extended) {
            return edge_op(ExtEdge{eidx, exvidx, j}, Transform::offdiagonal(value, grex::Scalar{}));
          } else {
            if (range.contains(j.index())) [[likely]] {
              return edge_op(Edge{eidx, vidx, index_convert(j, own_index_tag, dist_info)},
                             Transform::offdiagonal(value, grex::Scalar{}));
            }
            using Ret = decltype(edge_op(Edge{eidx, vidx, VertexIdx{}},
                                         Transform::offdiagonal(value, grex::Scalar{})));
            THES_RETURN_EMPTY_OPTIONAL(Ret);
          }
        };
        return row_.iterate(
          offdiag_op,
          [&]([[maybe_unused]] ExtVertexIdx j, auto value) THES_ALWAYS_INLINE {
            assert(j == exvidx);
            return vtx_op(Vertex{vidx}, Transform::diagonal(value, grex::Scalar{}));
          },
          offdiag_op, is_valued, is_ordered);
      }
    }
    THES_ALWAYS_INLINE
    constexpr decltype(auto) iterate_impl(auto vtx_op, auto edge_op, UnvaluedTag is_valued,
                                          AnyOrderingTag auto is_ordered,
                                          thes::AnyBoolTag auto is_extended) const {
      decltype(auto) dist_info = row_.distributed_info_storage();
      const VertexIdx vidx = row_.index();
      EdgeSize edge_idx = start_idx_;

      if constexpr (is_shared) {
        auto offdiag_op = [&](ExtVertexIdx j) THES_ALWAYS_INLINE {
          return edge_op(Edge{edge_idx++, vidx, index_convert(j, own_index_tag, dist_info)});
        };
        return row_.iterate(
          offdiag_op, [&](ExtVertexIdx j) THES_ALWAYS_INLINE { return vtx_op(Vertex{j}); },
          offdiag_op, is_valued, is_ordered);
      } else {
        const ExtVertexIdx exvidx = row_.ext_index();
        decltype(auto) range = dist_info.index_range_within(own_index_tag, local_index_tag);

        auto offdiag_op = [&](ExtVertexIdx j) THES_ALWAYS_INLINE {
          const auto eidx = edge_idx++;
          if constexpr (is_extended) {
            return edge_op(Edge{eidx, exvidx, j});
          } else {
            if (range.contains(j.index())) {
              return edge_op(Edge{eidx, vidx, index_convert(j, own_index_tag, dist_info)});
            }
            using Ret = decltype(edge_op(Edge{eidx, vidx, std::declval<VertexIdx>()}));
            THES_RETURN_EMPTY_OPTIONAL(Ret);
          }
        };
        return row_.iterate(
          offdiag_op,
          [&]([[maybe_unused]] ExtVertexIdx j) THES_ALWAYS_INLINE {
            assert(j == exvidx);
            return vtx_op(Vertex{vidx});
          },
          offdiag_op, is_valued, is_ordered);
      }
    }

    Row row_;
    EdgeSize start_idx_;
  };

private:
  struct VertexIterProvider {
    using Val = FullVertex;
    using Diff = decltype(std::declval<RowIter>() - std::declval<RowIter>());
    using IterTypes = thes::iter_provider::ValueTypes<Val, Diff>;

    static constexpr Val deref(const auto& self) {
      return FullVertex{*self.current_, *self.start_idx_};
    }
    static constexpr void incr(auto& self) {
      ++self.current_;
      ++self.start_idx_;
    }
    static constexpr void decr(auto& self) {
      --self.current_;
      --self.start_idx_;
    }
    static constexpr void iadd(auto& self, auto diff) {
      self.current_ += diff;
      self.start_idx_ += diff;
    }
    static constexpr void isub(auto& self, auto diff) {
      self.current_ -= diff;
      self.start_idx_ -= diff;
    }
    static constexpr bool eq(const auto& i1, const auto& i2) {
      return i1.current_ == i2.current_;
    }
    static constexpr std::strong_ordering three_way(const auto& i1, const auto& i2) {
      return i1.current_ <=> i2.current_;
    }
    static constexpr Diff sub(const auto& i1, const auto& i2) {
      return i1.current_ - i2.current_;
    }
  };

public:
  struct VertexIter : public thes::IteratorFacade<VertexIter, VertexIterProvider> {
    using DistInfo = thes::VoidConstLvalRef<DistributedInfo>;
    using DistInfoStorage = thes::VoidStorage<DistInfo>;

    friend struct VertexIterProvider;
    explicit VertexIter(RowIter&& current, EdgeIdxsIter start_idx)
        : current_(std::forward<RowIter>(current)), start_idx_(start_idx) {}

    VertexIter(const VertexIter&) = default;
    VertexIter& operator=(const VertexIter&) = default;
    VertexIter(VertexIter&&) noexcept = default;
    VertexIter& operator=(VertexIter&&) noexcept = default;

    ~VertexIter() = default;

  private:
    RowIter current_;
    EdgeIdxsIter start_idx_;
  };

  using const_iterator = VertexIter;

  MatrixGraph(TMat&& matrix, const auto& execution_policy)
      : matrix_(std::forward<TMat>(matrix)), start_(matrix_.row_num() + 1) {
    auto it = start_.begin();
    *it = EdgeSize{0};
    ++it;
    thes::transform_inclusive_scan(
      execution_policy, matrix_.begin(), matrix_.end(), it, std::plus{},
      [&](auto&& row) THES_ALWAYS_INLINE { return row.offdiagonal_num(); }, EdgeSize{0});
  }

  MatrixGraph(const MatrixGraph&) = delete;
  MatrixGraph& operator=(const MatrixGraph&) = delete;
  MatrixGraph(MatrixGraph&&) noexcept = default;
  MatrixGraph& operator=(MatrixGraph&&) noexcept = default;

  ~MatrixGraph() = default;

  [[nodiscard]] IsSymmetric is_value_symmetric() const {
    return matrix_.is_symmetric();
  }

  [[nodiscard]] const_iterator begin() const {
    return const_iterator(matrix_.begin(), start_.begin());
  }
  [[nodiscard]] const_iterator end() const {
    return const_iterator(matrix_.end(), start_.end());
  }

  [[nodiscard]] const Matrix& matrix() const {
    return matrix_;
  }

  [[nodiscard]] VertexSize vertex_num() const {
    return matrix_.row_num();
  }
  [[nodiscard]] VertexIdx vertex_index_begin() const {
    return VertexIdx{0};
  }
  [[nodiscard]] VertexIdx vertex_index_end() const {
    return VertexIdx{vertex_num()};
  }

  [[nodiscard]] EdgeSize max_edge_num() const {
    return start_[matrix_.row_num()];
  }
  [[nodiscard]] EdgeSize max_edge_index_end() const {
    return max_edge_num();
  }

  THES_ALWAYS_INLINE FullVertex full_vertex_at(VertexIdx index) const {
    return FullVertex(matrix_[index], start_[index_value(index)]);
  }
  THES_ALWAYS_INLINE FullVertex full_vertex_at(thes::AnyIndexPosition auto idx) const {
    return FullVertex(matrix_[idx],
                      start_[index_value(idx.index, own_index_tag, distributed_info_storage())]);
  }

  THES_ALWAYS_INLINE Weight head_vertex_weight(const ExtEdge& edge) const {
    return Transform::diagonal(matrix_.diagonal_at(edge.head().index()), grex::Scalar{});
  }

  std::optional<ExtEdge> reverse(const ExtEdge& edge) const {
    const ExtVertexIdx ext_tail_idx = edge.head().index();
    const ExtVertexIdx ext_head_idx = edge.tail().index();

    if constexpr (!is_shared) {
      if (!ext_tail_idx.is_convertible(own_index_tag, distributed_info())) {
        return ExtEdge{{}, ext_tail_idx, ext_head_idx};
      }
    }

    // The full row
    const VertexIdx tail_idx =
      index_convert(ext_tail_idx, own_index_tag, distributed_info_storage());
    const auto row = matrix_[tail_idx];

    std::optional<VertexSize> idx_in_row = row.offdiagonal_index_within(ext_head_idx);
    if (!idx_in_row.has_value()) {
      return std::nullopt;
    }
    return ExtEdge{start_[index_value(tail_idx)] + (*idx_in_row), ext_tail_idx, ext_head_idx};
  }

  decltype(auto) distributed_info_storage() const {
    return lineal::distributed_info_storage(matrix_);
  }
  const DistributedInfoStorage& distributed_info() const
  requires(!is_shared)
  {
    return matrix_.distributed_info();
  }

  Weight weight_of(const ExtEdge& edge) const {
    return Transform::offdiagonal(matrix_.entry_at(edge.tail().index(), edge.head().index()),
                                  grex::Scalar{});
  }

private:
  TMat matrix_;
  EdgeIdxs start_;
};

template<typename TTransform, typename TByteAlloc = thes::HugePagesAllocator<std::byte>,
         SharedMatrix TMat>
inline MatrixGraph<TMat, TTransform, TByteAlloc> make_matrix_graph(TMat&& matrix,
                                                                   const auto& execution_policy) {
  return {std::forward<TMat>(matrix), execution_policy};
}
} // namespace lineal::amg

#endif // INCLUDE_LINEAL_LINEAR_SOLVER_MULTIGRID_GRAPH_MATRIX_GRAPH_HPP
