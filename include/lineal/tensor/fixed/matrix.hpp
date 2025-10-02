// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_TENSOR_FIXED_MATRIX_HPP
#define INCLUDE_LINEAL_TENSOR_FIXED_MATRIX_HPP

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <type_traits>
#include <utility>

#include "thesauros/math.hpp"
#include "thesauros/static-ranges.hpp"
#include "thesauros/types.hpp"
#include "thesauros/utility.hpp"

#include "lineal/base/compat/operation.hpp"
#include "lineal/base/concept/scalar.hpp"
#include "lineal/base/enum.hpp"
#include "lineal/tensor/fixed/concepts.hpp"
#include "lineal/tensor/fixed/vector.hpp"

namespace lineal::fix {
struct MatrixDimensions {
  std::size_t row_num;
  std::size_t column_num;

  [[nodiscard]] constexpr bool is_square() const {
    return row_num == column_num;
  }

  constexpr bool operator==(const MatrixDimensions& other) const = default;
};

template<AnyMatrix TInner, std::size_t tCol>
struct FixedColumnView : public VectorBase {
  using Inner = std::decay_t<TInner>;
  using Value = Inner::Value;
  static constexpr std::size_t size = Inner::dimensions.row_num;
  static constexpr std::size_t column = tCol;

  explicit FixedColumnView(TInner&& inner) : inner_(std::forward<TInner>(inner)) {}

  template<std::size_t tIdx>
  requires(tIdx < size)
  Value operator[](thes::IndexTag<tIdx> /*idx*/) const {
    return inner_(thes::index_tag<tIdx>, thes::index_tag<column>);
  }

  FixedColumnView& operator*=(Value factor)
  requires(!std::is_const_v<TInner>)
  {
    thes::star::iota<0, size> |
      thes::star::for_each([&](auto i) { inner_(i, thes::index_tag<column>) *= factor; });
    return *this;
  }

private:
  TInner inner_;
};

template<AnyMatrix TInner>
struct DynamicColumnView : public VectorBase {
  using Inner = std::decay_t<TInner>;
  using Value = Inner::Value;
  static constexpr std::size_t size = Inner::dimensions.row_num;

  DynamicColumnView(TInner&& inner, std::size_t col)
      : inner_(std::forward<TInner>(inner)), col_(col) {}

  Value operator[](std::size_t idx) const {
    assert(idx < size);
    return inner_(idx, col_);
  }

  Value& operator[](std::size_t idx) {
    assert(idx < size);
    return inner_(idx, col_);
  }

  DynamicColumnView& operator*=(Value factor)
  requires(!std::is_const_v<TInner>)
  {
    thes::star::iota<0, size> |
      thes::star::for_each([&](auto i) { inner_(i.value, col_) *= factor; });
    return *this;
  }

  friend void swap(DynamicColumnView& lhs, DynamicColumnView& rhs) {
    thes::star::iota<0, size> | thes::star::for_each([&](auto i) {
      using std::swap;
      swap(lhs[i.value], rhs[i.value]);
    });
  }

  friend void swap(DynamicColumnView&& lhs, DynamicColumnView&& rhs) {
    swap(lhs, rhs);
  }

private:
  TInner inner_;
  std::size_t col_;
};

template<typename TFun, AnyMatrix... TInner>
struct TransformedMatrixView : public MatrixBase {
  using Inner = thes::Tuple<std::decay_t<TInner>...>;
  using Value =
    decltype(std::declval<TFun>()(std::declval<typename std::decay_t<TInner>::Value>()...));
  static constexpr MatrixDimensions dimensions =
    thes::star::unique_value(std::array{std::decay_t<TInner>::dimensions...}).value();

  struct Kind {
    static constexpr bool is_lower_triangular =
      (... && std::decay_t<TInner>::Kind::is_lower_triangular);
    static constexpr bool is_upper_triangular =
      (... && std::decay_t<TInner>::Kind::is_upper_triangular);
    static constexpr bool is_symmetric = (... && std::decay_t<TInner>::Kind::is_symmetric);
  };

  explicit constexpr TransformedMatrixView(TFun&& fun, TInner&&... inner)
      : fun_(std::forward<TFun>(fun)), inner_{std::forward<TInner>(inner)...} {}

  template<std::size_t tRow, std::size_t tCol>
  constexpr Value operator[](thes::IndexTag<tRow> row, thes::IndexTag<tCol> col) const {
    return thes::star::static_apply<sizeof...(TInner)>(
      [&]<std::size_t... tI>() { return fun_(get<tI>(inner_)[row, col]...); });
  }
  constexpr Value operator[](std::size_t row, std::size_t col) const {
    return thes::star::static_apply<sizeof...(TInner)>(
      [&]<std::size_t... tI>() { return fun_(get<tI>(inner_)[row, col]...); });
  }

private:
  [[no_unique_address]] TFun fun_;
  thes::Tuple<TInner...> inner_;
};

template<typename TFun, AnyMatrix... TMat>
constexpr auto transform(TFun&& fun, TMat&&... mat) {
  return TransformedMatrixView<TFun, TMat...>{std::forward<TFun>(fun), std::forward<TMat>(mat)...};
}

template<typename TValue, std::size_t tRows, std::size_t tCols>
struct DenseMatrix : public MatrixBase {
  using Value = TValue;
  static constexpr MatrixDimensions dimensions{.row_num = tRows, .column_num = tCols};
  static constexpr std::size_t element_num = tRows * tCols;
  using Data = std::array<Value, element_num>;

  struct Kind {};

  static constexpr DenseMatrix zero() {
    return DenseMatrix(thes::star::constant<element_num>(TValue{0}) | thes::star::to_array);
  }
  static constexpr DenseMatrix signaling_nan() {
    return DenseMatrix(
      thes::star::constant<element_num>(std::numeric_limits<TValue>::signaling_NaN()) |
      thes::star::to_array);
  }
  static constexpr DenseMatrix identity()
  requires(tRows == tCols)
  {
    return DenseMatrix{thes::star::index_transform<tRows * tCols>([&](auto idx) {
                         constexpr std::size_t i = idx / tCols;
                         constexpr std::size_t j = idx % tCols;
                         return TValue(i == j);
                       }) |
                       thes::star::to_array};
  }
  static constexpr DenseMatrix diagonal(const DenseVector<TValue, std::min(tRows, tCols)>& d) {
    return DenseMatrix{thes::star::index_transform<tRows * tCols>([&](auto idx) {
                         constexpr std::size_t i = idx / tCols;
                         constexpr std::size_t j = idx % tCols;
                         if constexpr (i == j) {
                           return d[thes::index_tag<i>];
                         } else {
                           return TValue{0};
                         }
                       }) |
                       thes::star::to_array};
  }

  explicit constexpr DenseMatrix(std::initializer_list<std::array<Value, tCols>> init)
      : data_{thes::star::index_transform<element_num>(
                [&](auto i) { return init.begin()[i / tCols][i % tCols]; }) |
              thes::star::to_array} {
    assert(init.size() == tRows);
  }

  template<AnyMatrix TMatrix>
  requires(TMatrix::dimensions == dimensions)
  explicit constexpr DenseMatrix(const TMatrix& other)
      : data_{thes::star::index_transform<0, element_num>([&](auto idx) {
                return other[thes::index_tag<idx / dimensions.column_num>,
                             thes::index_tag<idx % dimensions.column_num>];
              }) |
              thes::star::to_array} {}

  DenseMatrix() = default;
  template<AnyMatrix TMatrix>
  requires(TMatrix::dimensions == dimensions)
  DenseMatrix& operator=(const TMatrix& other) {
    thes::star::iota<0, dimensions.row_num> | thes::star::for_each([&](auto i) {
      thes::star::iota<0, dimensions.column_num> |
        thes::star::for_each([&](auto j) { (*this)[i, j] = Value(other[i, j]); });
    });
    return *this;
  }

  auto operator-() const {
    return transform([](Value s) { return -s; }, *this);
  }

  template<std::size_t tRow, std::size_t tCol>
  requires(tRow < dimensions.row_num && tCol < dimensions.column_num)
  constexpr Value operator[](thes::IndexTag<tRow> /*row*/, thes::IndexTag<tCol> /*col*/) const {
    return std::get<tRow * dimensions.column_num + tCol>(data_);
  }

  template<std::size_t tRow, std::size_t tCol>
  requires(tRow < dimensions.row_num && tCol < dimensions.column_num)
  constexpr Value& operator[](thes::IndexTag<tRow> /*row*/, thes::IndexTag<tCol> /*col*/) {
    return std::get<tRow * dimensions.column_num + tCol>(data_);
  }

  constexpr Value operator[](std::size_t row, std::size_t col) const {
    assert(row < dimensions.row_num && col < dimensions.column_num);
    return data_[row * dimensions.column_num + col];
  }
  constexpr Value& operator[](std::size_t row, std::size_t col) {
    assert(row < dimensions.row_num && col < dimensions.column_num);
    return data_[row * dimensions.column_num + col];
  }

  template<std::size_t tCol>
  requires(tCol < dimensions.column_num)
  [[nodiscard]] constexpr auto column(thes::IndexTag<tCol> /*col*/) const& {
    return FixedColumnView<const DenseMatrix&, tCol>{*this};
  }
  template<std::size_t tCol>
  requires(tCol < dimensions.column_num)
  [[nodiscard]] constexpr auto column(thes::IndexTag<tCol> /*col*/) & {
    return FixedColumnView<DenseMatrix&, tCol>{*this};
  }

  [[nodiscard]] constexpr auto column(const std::size_t col) const& {
    assert(col < dimensions.column_num);
    return DynamicColumnView<const DenseMatrix&>{*this, col};
  }
  [[nodiscard]] constexpr auto column(const std::size_t col) & {
    assert(col < dimensions.column_num);
    return DynamicColumnView<DenseMatrix&>{*this, col};
  }

  template<typename TOther>
  auto cast() const& {
    return transform([](Value s) { return static_cast<TOther>(s); }, *this);
  }

  const Value* data() const {
    return data_.data();
  }
  Value* data() {
    return data_.data();
  }

  // TODO This is public to make `DenseMatrix` structural, but if C++ ever allows instances of types
  //      with private members to be used as template parameters, this should really be private
  Data data_;

private:
  template<typename TOtherValue, std::size_t tOtherRows, std::size_t tOtherCols>
  friend struct DenseMatrix;
  explicit constexpr DenseMatrix(Data&& data) : data_(std::move(data)) {}
};

namespace trimpl {
template<TriangularKind tKind>
constexpr bool nonzero(std::size_t r, std::size_t c) {
  if constexpr (tKind == TriangularKind::lower) {
    return c <= r;
  } else {
    return r <= c;
  }
}

template<TriangularKind tKind, std::size_t tDim>
constexpr std::pair<std::size_t, std::size_t> index2pos(std::size_t i) {
  if constexpr (tKind == TriangularKind::lower) {
    const auto sqroot = thes::isqrt_floor<std::size_t>(8 * i + 1);
    const std::size_t row = (sqroot - 1) / 2;
    const std::size_t col = i - row * (row + 1) / 2;
    return std::make_pair(row, col);
  } else {
    constexpr std::size_t base = 2 * tDim + 1;
    const auto sqroot = thes::isqrt_ceil<std::size_t>(base * base - 8 * i);
    const std::size_t row = (base - sqroot) / 2;
    const std::size_t col = i - row * (2 * tDim - (row + 1)) / 2;
    return std::make_pair(row, col);
  }
}
template<TriangularKind tKind, std::size_t tDim>
constexpr std::size_t pos2index(std::size_t r, std::size_t c) {
  if constexpr (tKind == TriangularKind::lower) {
    return r * (r + 1) / 2 + c;
  } else {
    return r * (2 * tDim - (r + 1)) / 2 + c;
  }
}
} // namespace trimpl

template<typename TValue, std::size_t tDim, TriangularKind tKind>
struct DenseTriangularMatrix : public MatrixBase {
  using Value = TValue;
  static constexpr std::size_t dimension = tDim;
  static constexpr MatrixDimensions dimensions{.row_num = tDim, .column_num = tDim};
  static constexpr std::size_t element_num = (dimension * (dimension + 1)) / 2;
  using Data = std::array<Value, element_num>;
  using Pos = std::pair<std::size_t, std::size_t>;

  struct Kind {
    static constexpr bool is_lower_triangular = tKind == TriangularKind::lower;
    static constexpr bool is_upper_triangular = tKind == TriangularKind::upper;
  };

  static constexpr DenseTriangularMatrix zero() {
    return DenseTriangularMatrix(thes::star::constant<element_num>(TValue{0}) |
                                 thes::star::to_array);
  }

  template<typename... TArgs>
  requires(thes::TypeSeq<Value, std::decay_t<TArgs>...>::is_unique &&
           sizeof...(TArgs) == element_num)
  explicit constexpr DenseTriangularMatrix(TArgs&&... args) : data_{std::forward<TArgs>(args)...} {}

  template<AnyLowerTriangularMatrix TMatrix>
  requires(TMatrix::dimensions == dimensions)
  explicit constexpr DenseTriangularMatrix(const TMatrix& other)
      : data_{thes::star::index_transform<0, element_num>([&](auto idx) {
                constexpr auto pos = index2pos(idx);
                constexpr auto row = pos.first;
                constexpr auto col = pos.second;

                static_assert(row < dimension);
                static_assert(col <= row);
                static_assert((row * (row + 1)) / 2 + col == idx.value);

                return other[thes::index_tag<row>, thes::index_tag<col>];
              }) |
              thes::star::to_array} {}

  template<std::size_t tRow, std::size_t tCol>
  requires(tRow < dimension && tCol < dimension)
  constexpr Value operator[](thes::IndexTag<tRow> /*row*/, thes::IndexTag<tCol> /*col*/) const {
    if constexpr (trimpl::nonzero<tKind>(tRow, tCol)) {
      return std::get<pos2index(tRow, tCol)>(data_);
    } else {
      return Value{0};
    }
  }

  template<std::size_t tRow, std::size_t tCol>
  requires(tRow < dimension && tCol < dimension && trimpl::nonzero<tKind>(tRow, tCol))
  constexpr Value& operator[](thes::IndexTag<tRow> /*row*/, thes::IndexTag<tCol> /*col*/) {
    return std::get<pos2index(tRow, tCol)>(data_);
  }

  template<typename TOther>
  constexpr DenseTriangularMatrix<TOther, tDim, tKind> cast() const {
    return DenseTriangularMatrix<TOther, tDim, tKind>{
      data_ | thes::star::transform([](Value v) { return static_cast<TOther>(v); }) |
      thes::star::to_array};
  }

  static constexpr Pos index2pos(std::size_t i) {
    return trimpl::index2pos<tKind, dimension>(i);
  }

  static constexpr std::size_t pos2index(std::size_t r, std::size_t c) {
    return trimpl::pos2index<tKind, dimension>(r, c);
  }

private:
  explicit constexpr DenseTriangularMatrix(Data&& data) : data_(std::move(data)) {}

  Data data_;
};

template<typename TValue, std::size_t tDim, TriangularKind tKind>
struct TupleTriangularMatrix : public MatrixBase {
  using Value = TValue;
  static constexpr std::size_t dimension = tDim;
  static constexpr MatrixDimensions dimensions{.row_num = tDim, .column_num = tDim};
  static constexpr std::size_t element_num = (dimension * (dimension + 1)) / 2;
  using Data = thes::SizedTuple<Value, element_num>;
  using Pos = std::pair<std::size_t, std::size_t>;

  static constexpr TupleTriangularMatrix zero() {
    return TupleTriangularMatrix(thes::star::constant<element_num>(TValue{0}) |
                                 thes::star::to_tuple);
  }
  static constexpr TupleTriangularMatrix identity() {
    return TupleTriangularMatrix(thes::star::index_transform<element_num>([](auto i) {
                                   constexpr auto pos = index2pos(i);
                                   return Value(pos.first == pos.second);
                                 }) |
                                 thes::star::to_tuple);
  }

  constexpr TupleTriangularMatrix() = default;

  template<std::size_t tRow, std::size_t tCol>
  requires(tRow < dimension && tCol < dimension)
  constexpr Value operator[](thes::IndexTag<tRow> /*row*/, thes::IndexTag<tCol> /*col*/) const {
    if constexpr (trimpl::nonzero<tKind>(tRow, tCol)) {
      return get<pos2index(tRow, tCol)>(data_);
    } else {
      return Value{0};
    }
  }

  template<std::size_t tRow, std::size_t tCol>
  requires(tRow < dimension && tCol < dimension && trimpl::nonzero<tKind>(tRow, tCol))
  constexpr Value& operator[](thes::IndexTag<tRow> /*row*/, thes::IndexTag<tCol> /*col*/) {
    return get<pos2index(tRow, tCol)>(data_);
  }

  static constexpr Pos index2pos(std::size_t i) {
    return trimpl::index2pos<tKind, dimension>(i);
  }

  static constexpr std::size_t pos2index(std::size_t r, std::size_t c) {
    return trimpl::pos2index<tKind, dimension>(r, c);
  }

private:
  explicit constexpr TupleTriangularMatrix(Data&& data) : data_(std::move(data)) {}

  Data data_;
};

template<typename TInner>
struct LowerTriangularMatrixView : public MatrixBase {
  using Inner = std::decay_t<TInner>;
  using Value = Inner::Value;
  static constexpr MatrixDimensions dimensions = Inner::dimensions;
  static_assert(dimensions.row_num == dimensions.column_num, "Only square matrices are supported!");

  struct Kind {
    static constexpr bool is_lower_triangular = true;
    static constexpr bool is_upper_triangular = false;
    static constexpr bool is_symmetric = false;
  };

  explicit LowerTriangularMatrixView(TInner&& inner) : inner_(std::forward<TInner>(inner)) {}

  template<std::size_t tRow, std::size_t tCol>
  Value operator[](thes::IndexTag<tRow> row, thes::IndexTag<tCol> col) const {
    if constexpr (tRow < tCol) {
      return Value{0};
    } else {
      return inner_(row, col);
    }
  }

  template<std::size_t tRow, std::size_t tCol>
  Value& operator[](thes::IndexTag<tRow> row, thes::IndexTag<tCol> col) {
    if constexpr (tRow < tCol) {
      return Value{0};
    } else {
      return inner_(row, col);
    }
  }

private:
  TInner inner_;
};

template<AnyMatrix TMat>
constexpr auto lower_triangular_view(TMat&& mat) {
  return LowerTriangularMatrixView<TMat>{std::forward<TMat>(mat)};
}

template<AnyLowerTriangularMatrix TInner>
struct SymmetricMatrixView {
  using Inner = std::decay_t<TInner>;
  using Value = Inner::Value;
  static constexpr MatrixDimensions dimensions = Inner::dimensions;
  static_assert(dimensions.row_num == dimensions.column_num, "Only square matrices are supported!");

  struct Kind {
    static constexpr bool is_symmetric = true;
  };

  explicit SymmetricMatrixView(TInner&& inner) : inner_(std::forward<TInner>(inner)) {}

  template<std::size_t tRow, std::size_t tCol>
  Value operator[](thes::IndexTag<tRow> /*row*/, thes::IndexTag<tCol> /*col*/) const {
    return inner_(thes::index_tag<std::max(tRow, tCol)>, thes::index_tag<std::min(tRow, tCol)>);
  }

  template<std::size_t tRow, std::size_t tCol>
  Value& operator[](thes::IndexTag<tRow> /*row*/, thes::IndexTag<tCol> /*col*/) {
    return inner_(thes::index_tag<std::max(tRow, tCol)>, thes::index_tag<std::min(tRow, tCol)>);
  }

private:
  TInner inner_;
};

template<typename TInner>
struct TransposedMatrixView : public MatrixBase {
  using Inner = std::decay_t<TInner>;
  using Value = Inner::Value;
  static constexpr MatrixDimensions inner_dimensions = Inner::dimensions;
  static constexpr MatrixDimensions dimensions{
    .row_num = inner_dimensions.column_num,
    .column_num = inner_dimensions.row_num,
  };

  struct Kind {
    static constexpr bool is_lower_triangular = AnyUpperTriangularMatrix<Inner>;
    static constexpr bool is_upper_triangular = AnyLowerTriangularMatrix<Inner>;
    static constexpr bool is_symmetric = AnySymmetricMatrix<Inner>;
  };

  explicit constexpr TransposedMatrixView(TInner&& inner) : inner_(std::forward<TInner>(inner)) {}

  template<std::size_t tRow, std::size_t tCol>
  constexpr Value operator[](thes::IndexTag<tRow> row, thes::IndexTag<tCol> col) const {
    return inner_[col, row];
  }
  template<std::size_t tRow, std::size_t tCol>
  constexpr Value& operator[](thes::IndexTag<tRow> row, thes::IndexTag<tCol> col) {
    return inner_[col, row];
  }

  constexpr Value operator[](std::size_t row, std::size_t col) const {
    return inner_[col, row];
  }
  constexpr Value& operator[](std::size_t row, std::size_t col) {
    return inner_[col, row];
  }

private:
  TInner inner_;
};

template<AnyMatrix TMat>
constexpr auto transpose(TMat&& mat) {
  return TransposedMatrixView<TMat>{std::forward<TMat>(mat)};
}

template<AnyMatrix TLhs, AnyMatrix TRhs>
requires(TLhs::dimensions == TRhs::dimensions)
constexpr bool operator==(const TLhs& lhs, const TRhs& rhs) {
  for (std::size_t i = 0; i < lhs.dimensions.row_num; ++i) {
    for (std::size_t j = 0; j < lhs.dimensions.column_num; ++j) {
      if (lhs[i, j] != rhs[i, j]) {
        return false;
      }
    }
  }
  return true;
}

template<AnyMatrix TMat>
constexpr auto operator-(TMat&& mat) {
  return transform([](auto v) { return -v; }, std::forward<TMat>(mat));
}

template<AnyMatrix TLhs, IsScalar TRhs>
constexpr auto operator/(TLhs&& lhs, TRhs rhs) {
  return transform([rhs](auto a) { return a / rhs; }, std::forward<TLhs>(lhs));
}

#define LINEAL_FIX_DMAT_ARITH(OP) \
  template<AnyMatrix TLhs, AnyMatrix TRhs> \
  requires(TLhs::dimensions == TRhs::dimensions) \
  constexpr TLhs& operator OP##=(TLhs & lhs, const TRhs & rhs) { \
    thes::star::iota<0, TLhs::dimensions.row_num> | thes::star::for_each([&](auto i) { \
      thes::star::iota<0, TLhs::dimensions.column_num> | \
        thes::star::for_each([&](auto j) { lhs[i, j] OP## = rhs[i, j]; }); \
    }); \
    return lhs; \
  } \
  template<AnyMatrix TLhs, AnyMatrix TRhs> \
  requires(std::decay_t<TLhs>::dimensions == std::decay_t<TRhs>::dimensions) \
  constexpr auto operator OP(TLhs&& lhs, TRhs&& rhs) { \
    return transform([](auto a, auto b) { return a OP b; }, std::forward<TLhs>(lhs), \
                     std::forward<TRhs>(rhs)); \
  }

LINEAL_FIX_DMAT_ARITH(+)
LINEAL_FIX_DMAT_ARITH(-)
#undef LINEAL_FIX_DMAT_ARITH

template<typename TOther, AnyMatrix TMat>
constexpr auto cast(TMat&& mat) {
  return transform([](auto val) { return TOther(val); }, std::forward<TMat>(mat));
}

template<AnyMatrix TMat>
[[nodiscard]] constexpr bool is_finite(const TMat& mat) {
  bool finite = true;
  thes::star::iota<0, TMat::dimensions.row_num> | thes::star::for_each([&](auto i) {
    thes::star::iota<0, TMat::dimensions.column_num> |
      thes::star::for_each([&](auto j) { finite = finite && std::isfinite(mat[i, j]); });
  });
  return finite;
}

template<typename TReal, AnyMatrix TMat>
[[nodiscard]] constexpr TReal frobenius_squared(const TMat& mat) {
  TReal accum = 0;
  thes::star::iota<0, TMat::dimensions.row_num> | thes::star::for_each([&](auto i) {
    thes::star::iota<0, TMat::dimensions.column_num> |
      thes::star::for_each([&](auto j) { accum += TReal(mat[i, j]) * TReal(mat[i, j]); });
  });
  return accum;
}
template<typename TReal, AnyMatrix TMat>
[[nodiscard]] constexpr TReal frobenius_norm(const TMat& mat) {
  return compat::sqrt(frobenius_squared<TReal>(mat));
}
} // namespace lineal::fix

#endif // INCLUDE_LINEAL_TENSOR_FIXED_MATRIX_HPP
