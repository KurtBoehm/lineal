// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_IO_AGGREGATE_MAP_HPP
#define INCLUDE_LINEAL_IO_AGGREGATE_MAP_HPP

#include <type_traits>
#include <utility>
#include <vector>

#include "thesauros/format.hpp"
#include "thesauros/io.hpp"
#include "thesauros/ranges.hpp"
#include "thesauros/types.hpp"

#include "lineal/base.hpp"

namespace lineal {
template<typename TAggMap, bool tFormat, bool tOrdered, bool tIncludeForward>
struct AggregateMapPrinter {
  using AggMap = std::decay_t<TAggMap>;
  using Agg = AggMap::Aggregate;
  using Size = AggMap::Size;
  static constexpr auto fmt_tag = thes::FormattingTag<tFormat>{};

  explicit AggregateMapPrinter(TAggMap&& agg_map) : agg_map_(std::forward<TAggMap>(agg_map)) {}

  const AggMap& agg_map() const {
    return agg_map_;
  }

private:
  TAggMap agg_map_;
};

template<typename TAggMap, bool tFormat, bool tOrdered = true, bool tIncludeForward = true>
inline auto print_aggregate_map(TAggMap&& agg_map, thes::FormattingTag<tFormat> /*format*/,
                                OrderingTag<tOrdered> /*tag*/ = {},
                                thes::BoolTag<tIncludeForward> /*tag*/ = {}) {
  using Printer = AggregateMapPrinter<TAggMap, tFormat, tOrdered, tIncludeForward>;
  return Printer{std::forward<TAggMap>(agg_map)};
}
} // namespace lineal

template<typename TAggMap, bool tFormat, bool tOrdered, bool tIncludeForward>
struct fmt::formatter<lineal::AggregateMapPrinter<TAggMap, tFormat, tOrdered, tIncludeForward>>
    : public fmt::nested_formatter<typename std::decay_t<TAggMap>::Size> {
  using Self = lineal::AggregateMapPrinter<TAggMap, tFormat, tOrdered, tIncludeForward>;
  using Agg = Self::Agg;
  using Size = Self::Size;
  static constexpr auto fmt_tag = thes::FormattingTag<tFormat>{};

  auto format(const Self& printer, fmt::format_context& ctx) const {
    return this->write_padded(ctx, [&](auto it) {
      std::vector<Size> scrap{};
      const auto& agg_map = printer.agg_map();
      for (thes::Delimiter delim{"\n"}; const auto agg : thes::range(agg_map.coarse_row_num())) {
        decltype(auto) rows = [&]() -> decltype(auto) {
          if constexpr (tOrdered) {
            decltype(auto) fine_rows = agg_map.coarse_row_to_fine_rows(Agg{agg});
            scrap.resize(fine_rows.size());
            std::copy(fine_rows.begin(), fine_rows.end(), scrap.begin());
            std::sort(scrap.begin(), scrap.end());
            return scrap;
          } else {
            return agg_map.coarse_row_to_fine_rows(Agg{agg});
          }
        }();

        it = fmt::format_to(it, "{}", delim);
        it = thes::tsformat_to(fmt_tag, it, thes::rainbow_fg(agg), "{}: {}", agg, rows);
      }
      if constexpr (tIncludeForward) {
        *it++ = '\n';
        for (thes::Delimiter delim{", "}; const auto vtx : thes::range(agg_map.fine_row_num())) {
          const auto agg = agg_map[vtx];
          const auto format = agg.is_aggregate() ? thes::rainbow_fg(*agg) : fmt::text_style{};
          it = fmt::format_to(it, "{}", delim);
          it = thes::tsformat_to(fmt_tag, it, thes::rainbow_fg(agg), "{}: {}", vtx, agg);
        }
      }
      return it;
    });
  }
};

#endif // INCLUDE_LINEAL_IO_AGGREGATE_MAP_HPP
