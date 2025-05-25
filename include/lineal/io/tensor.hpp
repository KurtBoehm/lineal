// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_IO_TENSOR_HPP
#define INCLUDE_LINEAL_IO_TENSOR_HPP

#include <cstddef>
#include <filesystem>
#include <ranges>
#include <utility>

#include "thesauros/io.hpp"
#include "thesauros/macropolis.hpp"
#include "thesauros/math.hpp"
#include "thesauros/ranges.hpp"
#include "thesauros/types.hpp"
#include "thesauros/utility.hpp"

#include "lineal/base.hpp"

namespace lineal {
template<SharedMatrix TMat>
struct SmfReader {
  using Value = TMat::Value;
  using Size = TMat::Size;
  using NonZeroSize = TMat::NonZeroSize;

  static constexpr std::size_t data_off = 24;

  explicit SmfReader(std::filesystem::path path)
      : path_{std::move(path)}, reader_{path_}, row_num_{reader_.read(thes::type_tag<thes::u64>)},
        col_num_{reader_.read(thes::type_tag<thes::u64>)},
        nnz_num_{reader_.read(thes::type_tag<thes::u64>)} {}

  [[nodiscard]] thes::u64 row_num() const {
    return row_num_;
  }
  [[nodiscard]] thes::u64 col_num() const {
    return col_num_;
  }
  [[nodiscard]] thes::u64 nnz_num() const {
    return nnz_num_;
  }

  TMat read() && {
    using thes::f64;
    using thes::u64;

    typename TMat::RowWiseBuilder builder{};
    builder.initialize(*thes::safe_cast<Size>(row_num_), *thes::safe_cast<Size>(col_num_),
                       *thes::safe_cast<NonZeroSize>(nnz_num_), IsSymmetric{false});

    thes::FileReader entry_reader{path_};
    entry_reader.seek(*thes::safe_cast<long>(data_off + (row_num_ + 1) * 8), thes::Seek::set);
    u64 row_begin = reader_.read(thes::type_tag<u64>);
    u64 row_end = reader_.read(thes::type_tag<u64>);

    for (u64 row = 0; row < row_num_; ++row) {
      const u64 row_size = row_end - row_begin;
      for (u64 i = 0; i < row_size; ++i) {
        const u64 col = entry_reader.read(thes::type_tag<u64>);
        const f64 val = entry_reader.read(thes::type_tag<f64>);
        builder.insert(*thes::safe_cast<Size>(col), val);
      }
      ++builder;
      row_begin = row_end;
      row_end = reader_.read(thes::type_tag<u64>);
    }

    return std::move(builder).build();
  }

  TMat read(const Env auto& env) && {
    using thes::f64;
    using thes::u64;

    const auto size = *thes::safe_cast<Size>(row_num_);
    decltype(auto) expo = env.execution_policy();
    const std::size_t tnum = expo.thread_num();
    thes::UniformIndexSegmenter seg{size, tnum};

    typename TMat::Sizes row_nums(tnum);
    typename TMat::RowOffsets nnz_nums(tnum);
    u64 row_off_begin = reader_.read(thes::type_tag<u64>);
    for (const auto i : thes::range(tnum)) {
      const auto segment = seg.segment_range(i);
      row_nums[i] = segment.size();

      reader_.seek(*thes::safe_cast<long>(data_off + segment.end_value() * 8), thes::Seek::set);
      u64 row_off_end = reader_.read(thes::type_tag<u64>);
      nnz_nums[i] = *thes::safe_cast<NonZeroSize>(row_off_end - row_off_begin);
      row_off_begin = row_off_end;
    }

    auto builder = TMat::multithread_builder(thes::type_tag<Value>, lineal::unique_tag);
    builder.initialize(std::move(row_nums), std::move(nnz_nums), IsSymmetric{false}, thes::Empty{});

    expo.execute_segmented(
      size,
      [this, seg, &builder](std::size_t tidx, Size /*idx0*/, Size /*idx1*/) {
        const auto segment = seg.segment_range(tidx);
        auto thread_instance = builder.thread_instance(tidx);

        thes::FileReader row_off_reader{path_};
        row_off_reader.seek(*thes::safe_cast<long>(data_off + segment.begin_value() * 8),
                            thes::Seek::set);
        u64 row_begin = row_off_reader.read(thes::type_tag<u64>);
        u64 row_end = row_off_reader.read(thes::type_tag<u64>);

        thes::FileReader entry_reader{path_};
        entry_reader.seek(*thes::safe_cast<long>(data_off + (row_num_ + 1) * 8 + row_begin * 16),
                          thes::Seek::set);

        for ([[maybe_unused]] auto row : segment) {
          for ([[maybe_unused]] u64 i : thes::range(row_end - row_begin)) {
            const u64 col = entry_reader.read(thes::type_tag<u64>);
            const f64 val = entry_reader.read(thes::type_tag<f64>);
            thread_instance.insert(*thes::safe_cast<Size>(col), val);
          }
          ++thread_instance;
          row_begin = row_end;
          row_end = row_off_reader.read(thes::type_tag<u64>);
        }
      },
      tnum);

    return std::move(builder).build();
  }

private:
  std::filesystem::path path_;
  thes::FileReader reader_;
  thes::u64 row_num_;
  thes::u64 col_num_;
  thes::u64 nnz_num_;
};

struct SmfWriter {
  explicit SmfWriter(std::filesystem::path path) : path_{std::move(path)}, writer_{path_} {}

  template<SharedMatrix TMat>
  void write(const TMat& mat) && {
    using thes::f64;
    using thes::u64;

    constexpr bool has_non_zero_num = requires { mat.non_zero_num(); };

    writer_.write(u64{mat.row_num()});
    writer_.write(u64{mat.column_num()});
    if constexpr (has_non_zero_num) {
      writer_.write(u64{mat.non_zero_num()});
    } else {
      writer_.write(u64{0});
    }

    // Row offsets
    {
      u64 row_off = 0;
      writer_.write(row_off);
      for (decltype(auto) row : mat) {
        row.iterate(
          [&](auto /*j*/, auto val) THES_ALWAYS_INLINE {
            if (val != 0) {
              ++row_off;
            }
          },
          valued_tag, ordered_tag);
        writer_.write(row_off);
      }
      if constexpr (!has_non_zero_num) {
        auto pos = writer_.tell();
        writer_.seek(16, thes::Seek::set);
        writer_.write(row_off);
        writer_.seek(pos, thes::Seek::set);
      }
    }
    // Column indices and values are interleaved
    {
      for (auto row : mat) {
        row.iterate(
          [&](auto j, auto val) THES_ALWAYS_INLINE {
            if (val != 0) {
              writer_.write(u64{j});
              writer_.write(f64{val});
            }
          },
          valued_tag, ordered_tag);
      }
    }
  }

private:
  std::filesystem::path path_;
  thes::FileWriter writer_;
};

template<SharedVector TVec>
struct DvfReader {
  using Value = TVec::Value;
  using Size = TVec::Size;

  static constexpr std::size_t data_off = 8;

  explicit DvfReader(std::filesystem::path path)
      : path_{std::move(path)}, reader_{path_}, size_{reader_.read(thes::type_tag<thes::u64>)} {}

  [[nodiscard]] thes::u64 size() const {
    return size_;
  }

  TVec read() && {
    TVec vec(*thes::safe_cast<Size>(size_));
    for (auto& v : vec) {
      v = reader_.read(thes::type_tag<thes::f64>);
    }
    return vec;
  }

  TVec read(const Env auto& env) && {
    const auto size = *thes::safe_cast<Size>(size_);
    decltype(auto) expo = env.execution_policy();
    const std::size_t tnum = expo.thread_num();

    TVec vec(size);
    expo.execute_segmented(
      size,
      [this, &vec](std::size_t /*tidx*/, Size idx0, Size idx1) {
        thes::FileReader thread_reader{path_};
        thread_reader.seek(*thes::safe_cast<long>(data_off + idx0 * 8), thes::Seek::set);

        for (auto& v : std::ranges::subrange(vec.begin() + idx0, vec.begin() + idx1)) {
          v = thread_reader.read(thes::type_tag<thes::f64>);
        }
      },
      tnum);

    return vec;
  }

private:
  std::filesystem::path path_;
  thes::FileReader reader_;
  thes::u64 size_;
};

struct DvfWriter {
  explicit DvfWriter(std::filesystem::path path) : path_{std::move(path)}, writer_{path_} {}

  template<SharedVector TVec>
  void write(const TVec& vec) && {
    using thes::f64;
    using thes::u64;

    writer_.write(u64{vec.size()});
    for (auto v : vec) {
      writer_.write(f64{v});
    }
  }

private:
  std::filesystem::path path_;
  thes::FileWriter writer_;
};
} // namespace lineal

#endif // INCLUDE_LINEAL_IO_TENSOR_HPP
