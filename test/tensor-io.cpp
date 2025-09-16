// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstdio>
#include <filesystem>
#include <optional>
#include <type_traits>
#include <utility>

#include "thesauros/containers.hpp"
#include "thesauros/execution.hpp"
#include "thesauros/format.hpp"
#include "thesauros/io.hpp"
#include "thesauros/static-ranges.hpp"
#include "thesauros/test.hpp"
#include "thesauros/types.hpp"
#include "thesauros/utility.hpp"

#include "lineal/lineal.hpp"

using Real = double;
using SizeByte = thes::ByteInteger<4>;
using Size = SizeByte::Unsigned;
using Mat = lineal::CsrMatrix<Real, SizeByte, thes::ByteInteger<5>>;
using Vec = lineal::DenseVector<Real, Size>;

template<typename TThreadNum = thes::Empty, thes::AnyBoolTag TTiling = thes::TrueTag>
static auto make_expo(TThreadNum thread_num = {}, TTiling tiling = {}) {
  auto mkexec = [&] {
    if constexpr (std::same_as<TThreadNum, thes::Empty>) {
      return thes::SequentialExecutor{};
    } else {
      return thes::FixedStdThreadPool(thread_num);
    }
  };

  using Executor = decltype(mkexec());
  using TileSizes =
    std::conditional_t<tiling, thes::star::Constant<3, thes::ValueTag<Size, 8>>, void>;
  using ExPo =
    lineal::AdaptivePolicy<Executor, TileSizes, grex::full_tag<grex::native_sizes<Real>.back()>>;
  return ExPo{std::in_place, mkexec, std::nullopt};
}

template<typename TThreadNum = thes::Empty>
static auto make_env(TThreadNum thread_num = {}) {
  auto mkexpo = [&] { return make_expo(thread_num); };
  return lineal::Environment{std::in_place, mkexpo,
                             lineal::JsonLogger{thes::Indentation{2}, lineal::unflush_tag}};
}

int main(int /*argc*/, const char** const argv) {
  const auto base_path = std::filesystem::canonical(std::filesystem::path{argv[0]}.parent_path());
  const auto lhs_path = base_path / "UFZ_CT_02_LHS.smf";
  const auto rhs_path = base_path / "UFZ_CT_02_RHS.dvf";

  auto env = make_env(12U);

  for (bool is_seq : {true, false}) {
    auto envi = env.add_object(is_seq ? "lhs_seq" : "lhs_par");
    auto mat = [&] {
      lineal::SmfReader<Mat> reader{lhs_path};
      envi.log("dimensions",
               fmt::format("{}×{}@{}", reader.row_num(), reader.col_num(), reader.nnz_num()));
      if (is_seq) {
        return std::move(reader).read();
      }
      return std::move(reader).read(envi);
    }();

    std::filesystem::path tmp_path{std::tmpnam(nullptr)};
    lineal::SmfWriter{tmp_path}.write(mat);

    auto buf1 = thes::FileReader{lhs_path}.read_full(thes::type_tag<thes::DynamicBuffer>);
    auto buf2 = thes::FileReader{tmp_path}.read_full(thes::type_tag<thes::DynamicBuffer>);
    THES_ALWAYS_ASSERT(buf1.size() == buf2.size());
    THES_ALWAYS_ASSERT(std::equal(buf1.span().begin(), buf1.span().end(), buf2.span().begin()));
  }

  for (bool is_seq : {true, false}) {
    auto envi = env.add_object(is_seq ? "rhs_seq" : "rhs_par");
    lineal::DvfReader<Vec> reader{rhs_path};
    envi.log("size", reader.size());
    auto rhs = std::move(reader).read();

    std::filesystem::path tmp_path{std::tmpnam(nullptr)};
    lineal::DvfWriter{tmp_path}.write(rhs);

    auto buf1 = thes::FileReader{rhs_path}.read_full(thes::type_tag<thes::DynamicBuffer>);
    auto buf2 = thes::FileReader{tmp_path}.read_full(thes::type_tag<thes::DynamicBuffer>);
    THES_ALWAYS_ASSERT(buf1.size() == buf2.size());
    THES_ALWAYS_ASSERT(std::equal(buf1.span().begin(), buf1.span().end(), buf2.span().begin()));
  }
}
