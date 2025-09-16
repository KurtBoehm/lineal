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
#include <type_traits>
#include <utility>

#include "thesauros/math.hpp"
#include "thesauros/static-ranges.hpp"
#include "thesauros/types.hpp"

#include "lineal/tensor/fixed/concepts.hpp"

namespace lineal::fix {
struct MatrixDimensions {
  std::size_t row_num;
  std::size_t column_num;

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

template<AnyMatrix TInner, typename TFun>
struct TransformedMatrixView : public MatrixBase {
  using Inner = std::decay_t<TInner>;
  using Value =
    decltype(std::declval<TFun>()(std::declval<typename std::decay_t<TInner>::Value>()));
  static constexpr MatrixDimensions dimensions = Inner::dimensions;

  using Kind = Inner::Kind;

  explicit TransformedMatrixView(TInner&& inner, TFun&& fun)
      : inner_(std::forward<TInner>(inner)), fun_(std::forward<TFun>(fun)) {}

  template<std::size_t tRow, std::size_t tCol>
  Value operator()(thes::IndexTag<tRow> row, thes::IndexTag<tCol> col) const {
    return fun_(inner_(row, col));
  }

private:
  TInner inner_;
  [[no_unique_address]] TFun fun_;
};

template<AnyMatrix TMat, typename TFun>
constexpr auto transform(TMat&& mat, TFun&& fun) {
  return TransformedMatrixView<TMat, TFun>{std::forward<TMat>(mat), std::forward<TFun>(fun)};
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
                return other(thes::index_tag<idx / dimensions.column_num>,
                             thes::index_tag<idx % dimensions.column_num>);
              }) |
              thes::star::to_array} {}

  DenseMatrix() = default;
  template<AnyMatrix TMatrix>
  requires(TMatrix::dimensions == dimensions)
  DenseMatrix& operator=(const TMatrix& other) {
    thes::star::iota<0, dimensions.row_num> | thes::star::for_each([&](auto i) {
      thes::star::iota<0, dimensions.column_num> |
        thes::star::for_each([&](auto j) { (*this)(i, j) = other(i, j); });
    });
    return *this;
  }

  template<std::size_t tRow, std::size_t tCol>
  requires(tRow < dimensions.row_num && tCol < dimensions.column_num)
  constexpr Value operator()(thes::IndexTag<tRow> /*row*/, thes::IndexTag<tCol> /*col*/) const {
    return std::get<tRow * dimensions.column_num + tCol>(data_);
  }

  template<std::size_t tRow, std::size_t tCol>
  requires(tRow < dimensions.row_num && tCol < dimensions.column_num)
  constexpr Value& operator()(thes::IndexTag<tRow> /*row*/, thes::IndexTag<tCol> /*col*/) {
    return std::get<tRow * dimensions.column_num + tCol>(data_);
  }

  constexpr Value operator()(std::size_t row, std::size_t col) const {
    assert(row < dimensions.row_num && col < dimensions.column_num);
    return data_[row * dimensions.column_num + col];
  }
  constexpr Value& operator()(std::size_t row, std::size_t col) {
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
    return transform(*this, [](Value s) { return static_cast<TOther>(s); });
  }

  const Value* data() const {
    return data_.data();
  }
  Value* data() {
    return data_.data();
  }

private:
  template<typename TOtherValue, std::size_t tOtherRows, std::size_t tOtherCols>
  friend struct DenseMatrix;
  explicit constexpr DenseMatrix(Data&& data) : data_(std::move(data)) {}

  Data data_;
};

template<typename TValue, std::size_t tDim>
struct DenseLowerTriangularMatrix : public MatrixBase {
  using Value = TValue;
  static constexpr std::size_t dimension = tDim;
  static constexpr MatrixDimensions dimensions{.row_num = tDim, .column_num = tDim};
  static constexpr std::size_t element_num = (dimension * (dimension + 1)) / 2;
  using Data = std::array<Value, element_num>;

  struct Kind {
    static constexpr bool is_lower_triangular = true;
  };

  static constexpr DenseLowerTriangularMatrix zero() {
    return DenseLowerTriangularMatrix(thes::star::constant<element_num>(TValue{0}) |
                                      thes::star::to_array);
  }

  template<typename... TArgs>
  requires(thes::TypeSeq<Value, std::decay_t<TArgs>...>::is_unique &&
           sizeof...(TArgs) == element_num)
  explicit DenseLowerTriangularMatrix(TArgs&&... args) : data_{std::forward<TArgs>(args)...} {}

  template<AnyLowerTriangularMatrix TMatrix>
  requires(TMatrix::dimensions == dimensions)
  explicit DenseLowerTriangularMatrix(const TMatrix& other)
      : data_{thes::star::index_transform<0, element_num>([&](auto idx) {
                constexpr auto sqroot = thes::int_root<2, std::size_t>(8 * idx.value + 1);
                constexpr auto row = (sqroot - 1) / 2;
                constexpr auto col = idx.value - (row * (row + 1)) / 2;

                static_assert(row < dimension);
                static_assert(col <= row);
                static_assert((row * (row + 1)) / 2 + col == idx.value);

                return other(thes::index_tag<row>, thes::index_tag<col>);
              }) |
              thes::star::to_array} {}

  template<std::size_t tRow, std::size_t tCol>
  requires(tRow < dimension && tCol < dimension)
  Value operator()(thes::IndexTag<tRow> /*row*/, thes::IndexTag<tCol> /*col*/) const {
    if constexpr (tCol <= tRow) {
      return std::get<(tRow * (tRow + 1)) / 2 + tCol>(data_);
    } else {
      return Value{0};
    }
  }

  template<std::size_t tRow, std::size_t tCol>
  requires(tRow < dimension && tCol < dimension && tRow >= tCol)
  Value& operator()(thes::IndexTag<tRow> /*row*/, thes::IndexTag<tCol> /*col*/) {
    return std::get<(tRow * (tRow + 1)) / 2 + tCol>(data_);
  }

  template<typename TOther>
  DenseLowerTriangularMatrix<TOther, tDim> cast() const {
    return DenseLowerTriangularMatrix<TOther, tDim>{
      data_ | thes::star::transform([](Value v) { return static_cast<TOther>(v); }) |
      thes::star::to_array};
  }

private:
  explicit DenseLowerTriangularMatrix(Data&& data) : data_(std::move(data)) {}

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
  Value operator()(thes::IndexTag<tRow> row, thes::IndexTag<tCol> col) const {
    if constexpr (tRow < tCol) {
      return Value{0};
    } else {
      return inner_(row, col);
    }
  }

  template<std::size_t tRow, std::size_t tCol>
  Value& operator()(thes::IndexTag<tRow> row, thes::IndexTag<tCol> col) {
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
  Value operator()(thes::IndexTag<tRow> /*row*/, thes::IndexTag<tCol> /*col*/) const {
    return inner_(thes::index_tag<std::max(tRow, tCol)>, thes::index_tag<std::min(tRow, tCol)>);
  }

  template<std::size_t tRow, std::size_t tCol>
  Value& operator()(thes::IndexTag<tRow> /*row*/, thes::IndexTag<tCol> /*col*/) {
    return inner_(thes::index_tag<std::max(tRow, tCol)>, thes::index_tag<std::min(tRow, tCol)>);
  }

  template<typename TOther>
  auto cast() const& {
    return transform(*this, [](Value s) { return static_cast<TOther>(s); });
  }

private:
  TInner inner_;
};

template<typename TInner>
struct TransposedMatrixView {
  using Inner = std::decay_t<TInner>;
  using Scalar = Inner::Scalar;
  static constexpr MatrixDimensions inner_dimensions = Inner::dimensions.column_num;
  static constexpr MatrixDimensions dimensions{
    .row_num = inner_dimensions.column_num,
    .column_num = inner_dimensions.row_num,
  };

  struct Kind {
    static constexpr bool is_lower_triangular = AnyUpperTriangularMatrix<Inner>;
    static constexpr bool is_upper_triangular = AnyLowerTriangularMatrix<Inner>;
    static constexpr bool is_symmetric = AnySymmetricMatrix<Inner>;
  };

  explicit TransposedMatrixView(TInner&& inner) : inner_(std::forward<TInner>(inner)) {}

  template<std::size_t tRow, std::size_t tCol>
  Scalar operator()(thes::IndexTag<tRow> row, thes::IndexTag<tCol> col) const {
    return inner_(col, row);
  }

  template<std::size_t tRow, std::size_t tCol>
  Scalar& operator()(thes::IndexTag<tRow> row, thes::IndexTag<tCol> col) {
    return inner_(col, row);
  }

private:
  TInner inner_;
};

template<AnyMatrix TMat>
constexpr auto transposed(TMat&& mat) {
  return TransposedMatrixView<TMat>{std::forward<TMat>(mat)};
}
} // namespace lineal::fix

#endif // INCLUDE_LINEAL_TENSOR_FIXED_MATRIX_HPP
