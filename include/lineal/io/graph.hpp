// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_IO_GRAPH_HPP
#define INCLUDE_LINEAL_IO_GRAPH_HPP

#include <concepts>
#include <utility>

#include "thesauros/format.hpp"
#include "thesauros/io.hpp"

#include "lineal/base.hpp"
#include "lineal/linear-solver/multigrid.hpp"
#include "lineal/parallel/def.hpp"

namespace lineal {
template<SharedGraph TGraph, bool tSkipZero, bool tFormat, bool tIsOrdered>
struct WeightedGraphPrinter {
  static constexpr auto fmt_tag = thes::FormattingTag<tFormat>{};

  TGraph graph;
};

template<SharedGraph TGraph, bool tSkipZero, bool tFormat, bool tIsOrdered>
[[nodiscard]] inline auto
weighted_graph_print(TGraph&& graph, ZeroSkippingTag<tSkipZero> /*skip_zero*/,
                     thes::FormattingTag<tFormat> /*format*/, OrderingTag<tIsOrdered> /*ordered*/) {
  return WeightedGraphPrinter<TGraph, tSkipZero, tFormat, tIsOrdered>{std::forward<TGraph>(graph)};
}

template<SharedGraph TGraph, bool tSkipZero, bool tFormat, bool tIsOrdered>
struct PropertiedGraphPrinter {
  static constexpr auto fmt_tag = thes::FormattingTag<tFormat>{};

  TGraph graph;
};

template<SharedGraph TGraph, bool tSkipZero, bool tFormat, bool tIsOrdered>
[[nodiscard]] inline auto propertied_graph_print(TGraph&& graph,
                                                 ZeroSkippingTag<tSkipZero> /*skip_zero*/,
                                                 thes::FormattingTag<tFormat> /*format*/,
                                                 OrderingTag<tIsOrdered> /*ordered*/) {
  return PropertiedGraphPrinter<TGraph, tSkipZero, tFormat, tIsOrdered>{
    std::forward<TGraph>(graph)};
}
} // namespace lineal

template<>
struct fmt::formatter<lineal::amg::EdgeProperties> : public thes::SimpleFormatter<> {
  auto format(lineal::amg::EdgeProperties props, fmt::format_context& ctx) const {
    return this->write_padded(ctx, [&](auto it) {
      return fmt::format_to(it, "{:d}{:d}", props.influences(), props.depends());
    });
  }
};

template<typename TVtxIdx, std::unsigned_integral TEdgeSize>
struct fmt::formatter<lineal::amg::Edge<TVtxIdx, TEdgeSize>>
    : public fmt::nested_formatter<TVtxIdx> {
  auto format(lineal::amg::Edge<TVtxIdx, TEdgeSize> edge, fmt::format_context& ctx) const {
    return this->write_padded(ctx, [&](auto it) {
      return fmt::format_to(it, "[{}→{}]", this->nested(edge.tail_idx()),
                            this->nested(edge.head_idx()));
    });
  }
};

template<typename TSizeByte, lineal::OptIndexTag TIdxTag>
struct fmt::formatter<lineal::amg::Aggregate<TSizeByte, TIdxTag>>
    : public fmt::nested_formatter<typename TSizeByte::Unsigned> {
  auto format(lineal::amg::Aggregate<TSizeByte, TIdxTag> agg, fmt::format_context& ctx) const {
    return this->write_padded(ctx, [&](auto it) {
      if (agg.is_unaggregated()) {
        return fmt::format_to(it, "???");
      }
      if (agg.is_isolated()) {
        return fmt::format_to(it, "iso");
      }
      return fmt::format_to(it, "{}", this->nested(agg.index_));
    });
  }
};

template<lineal::SharedGraph TGraph, bool tSkipZero, bool tFormat, bool tIsOrdered>
struct fmt::formatter<lineal::WeightedGraphPrinter<TGraph, tSkipZero, tFormat, tIsOrdered>>
    : public thes::SimpleFormatter<> {
  using Self = lineal::WeightedGraphPrinter<TGraph, tSkipZero, tFormat, tIsOrdered>;
  static constexpr auto fmt_tag = thes::FormattingTag<tFormat>{};

  auto format(const Self& printer, fmt::format_context& ctx) const {
    return this->write_padded(ctx, [&](auto it) {
      for (thes::Delimiter vtx_delim{"\n"}; auto vertex : printer.graph) {
        it = fmt::format_to(it, "{}{}: ", vtx_delim,
                            thes::tstyled(fmt_tag, vertex.index(), thes::fg_green));

        thes::Delimiter delim{", "};
        vertex.iterate(
          [&](auto vtx, auto val) {
            it = fmt::format_to(it, "{}", delim);
            it = thes::tsformat_to(fmt_tag, it, thes::fg_blue, "{}:{}", vtx.index(), val);
          },
          [&](auto edge, auto val) {
            if constexpr (tSkipZero) {
              if (val == 0) {
                return;
              }
            }
            it = fmt::format_to(it, "{}", delim);
            it = thes::tsformat_to(fmt_tag, it, thes::fg_yellow, "{}@{}:{}", edge.head().index(),
                                   edge.index(), val);
          },
          lineal::valued_tag, lineal::OrderingTag<tIsOrdered>{});
      }

      return it;
    });
  }
};

template<lineal::SharedGraph TGraph, bool tSkipZero, bool tFormat, bool tIsOrdered>
struct fmt::formatter<lineal::PropertiedGraphPrinter<TGraph, tSkipZero, tFormat, tIsOrdered>>
    : public thes::SimpleFormatter<> {
  using Self = lineal::PropertiedGraphPrinter<TGraph, tSkipZero, tFormat, tIsOrdered>;
  static constexpr auto fmt_tag = thes::FormattingTag<tFormat>{};

  auto format(const Self& printer, fmt::format_context& ctx) const {
    return this->write_padded(ctx, [&](auto it) {
      for (thes::Delimiter vtx_delim{"\n"}; auto vertex : printer.graph) {
        const auto iso = printer.graph.is_isolated(vertex.vertex());
        it = fmt::format_to(it, "{}", vtx_delim);
        it = thes::tsformat_to(fmt_tag, it, iso ? thes::fg_red : thes::fg_green, "{}",
                               vertex.index(), iso);
        it = fmt::format_to(it, ": ");

        thes::Delimiter delim{", "};
        vertex.iterate(
          [&](auto vtx, auto val) {
            it = fmt::format_to(it, "{}", delim);
            it = thes::tsformat_to(fmt_tag, it, thes::fg_blue, "{}:{}", vtx.index(), val);
          },
          [&](auto edge, auto val) {
            if constexpr (tSkipZero) {
              if (val == 0) {
                return;
              }
            }
            it = fmt::format_to(it, "{}", delim);
            it = thes::tsformat_to(fmt_tag, it, thes::fg_yellow, "{}:{}:{}", edge.head().index(),
                                   val, printer.graph.get_edge_properties(edge));
          },
          lineal::valued_tag, lineal::OrderingTag<tIsOrdered>{});
      }

      return it;
    });
  }
};

#endif // INCLUDE_LINEAL_IO_GRAPH_HPP
