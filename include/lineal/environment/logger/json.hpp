// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_ENVIRONMENT_LOGGER_JSON_HPP
#define INCLUDE_LINEAL_ENVIRONMENT_LOGGER_JSON_HPP

#include <chrono>
#include <cstdio>
#include <string_view>

#include "thesauros/format.hpp"
#include "thesauros/io.hpp"
#include "thesauros/macropolis.hpp"
#include "thesauros/types.hpp"
#include "thesauros/utility.hpp"

#include "lineal/base/tag.hpp"
#include "lineal/io/synchronized.hpp"

namespace lineal {
namespace detail {
enum struct LoggerLevelKind : thes::u8 { OBJECT, ARRAY, NAMED_OBJECT };

template<AnyFlushingTag auto tFlush>
struct BaseLogger {
  BaseLogger() = default;
  explicit BaseLogger(thes::Indentation indent) : indent_outer{indent}, indent_inner{indent + 1} {}

  void separate() const {
    sync_print("{}{}{}", delimiter, indent_inner.separator(), indent_inner);
    flush();
  }

  static inline void flush() {
    if constexpr (tFlush) {
      (void)std::fflush(stdout);
    }
  }

  thes::Delimiter delimiter{", "};
  thes::Indentation indent_outer{};
  thes::Indentation indent_inner{};
};

template<AnyFlushingTag auto tFlush, LoggerLevelKind tKind = LoggerLevelKind::OBJECT>
struct JsonLogger;

template<AnyFlushingTag auto tFlush>
struct ObjectLogger : public BaseLogger<tFlush> {
  ObjectLogger() = default;
  explicit ObjectLogger(thes::Indentation indent) : BaseLogger<tFlush>{indent} {};

  ObjectLogger(const ObjectLogger&) = delete;
  ObjectLogger(ObjectLogger&& other) noexcept = default;
  ObjectLogger& operator=(const ObjectLogger&) = delete;
  ObjectLogger& operator=(ObjectLogger&&) = delete;
  ~ObjectLogger() = default;

  void destruct() const {
    if (is_unmoved()) {
      const auto dur = std::chrono::duration<double>(Clock::now() - creation_time_).count();
      log("overall_duration", dur);
    }
  }

  [[nodiscard]] JsonLogger<tFlush, LoggerLevelKind::ARRAY> add_array(const auto& key) const;
  [[nodiscard]] JsonLogger<tFlush, LoggerLevelKind::OBJECT> add_object(const auto& key) const;

  void log(std::string_view key, const auto& value) const {
    this->separate();
    sync_print("{}: {}", thes::json_print(key), thes::json_print(value, this->indent_inner));
  }

  template<typename T>
  void log_named(const T& value) const {
    using Info = thes::TypeInfo<T>;
    log(Info::serial_name.view(), value);
  }

  [[nodiscard]] bool is_unmoved() const {
    return detector_.is_unmoved();
  }

private:
  using Clock = std::chrono::steady_clock;
  Clock::time_point creation_time_{Clock::now()};
  thes::MoveDetector detector_{};
};

template<AnyFlushingTag auto tFlush>
struct JsonLogger<tFlush, LoggerLevelKind::OBJECT> : ObjectLogger<tFlush> {
  explicit JsonLogger(bool outer) : JsonLogger{outer, {}} {}
  JsonLogger(bool outer, thes::Indentation indent) : ObjectLogger<tFlush>{indent}, outer_{outer} {
    sync_print("{{");
  };

  JsonLogger(const JsonLogger&) = delete;
  JsonLogger(JsonLogger&& other) noexcept = default;
  JsonLogger& operator=(const JsonLogger&) = delete;
  JsonLogger& operator=(JsonLogger&&) = delete;

  ~JsonLogger() {
    this->destruct();
    if (this->is_unmoved()) {
      sync_print("{}{}}}", this->indent_outer.separator(), this->indent_outer);
      this->flush();
      if (outer_) {
        sync_print("\n");
        this->flush();
      }
    }
  }

private:
  bool outer_;
};

template<AnyFlushingTag auto tFlush>
struct JsonLogger<tFlush, LoggerLevelKind::NAMED_OBJECT> : ObjectLogger<tFlush> {
  explicit JsonLogger(std::string_view name) : JsonLogger{name, {}} {}
  JsonLogger(std::string_view name, thes::Indentation indent)
      : ObjectLogger<tFlush>{indent + 1}, indent_base_{indent} {
    sync_print("{{{}{}{}: {{", this->indent_outer.separator(), this->indent_outer,
               thes::json_print(name));
    this->flush();
  };

  JsonLogger(const JsonLogger&) = delete;
  JsonLogger(JsonLogger&& other) noexcept = default;
  JsonLogger& operator=(const JsonLogger&) = delete;
  JsonLogger& operator=(JsonLogger&&) = delete;

  ~JsonLogger() {
    this->destruct();
    if (this->is_unmoved()) {
      sync_print("{}{}}}{}{}}}", this->indent_outer.separator(), this->indent_outer,
                 this->indent_base_.separator(), indent_base_);
      this->flush();
    }
  }

private:
  thes::Indentation indent_base_;
};

template<AnyFlushingTag auto tFlush>
struct JsonLogger<tFlush, LoggerLevelKind::ARRAY> : BaseLogger<tFlush> {
  JsonLogger() : JsonLogger{{}} {}
  explicit JsonLogger(thes::Indentation indent) : BaseLogger<tFlush>{indent} {
    sync_print("[");
  };

  JsonLogger(const JsonLogger&) = delete;
  JsonLogger(JsonLogger&& other) noexcept = default;
  JsonLogger& operator=(const JsonLogger&) = delete;
  JsonLogger& operator=(JsonLogger&&) = delete;

  ~JsonLogger() {
    if (det_.is_unmoved()) {
      sync_print("{}{}]", this->indent_outer.separator(), this->indent_outer);
      this->flush();
    }
  }

  [[nodiscard]] JsonLogger<tFlush, LoggerLevelKind::OBJECT> add_object() const {
    this->separate();
    return {false, this->indent_inner};
  }
  [[nodiscard]] JsonLogger<tFlush, LoggerLevelKind::NAMED_OBJECT>
  add_object(const auto& key) const {
    this->separate();
    return {key, this->indent_inner};
  }

private:
  thes::MoveDetector det_{};
};

template<AnyFlushingTag auto tFlush>
template<typename T>
[[nodiscard]] JsonLogger<tFlush, LoggerLevelKind::OBJECT>
ObjectLogger<tFlush>::add_object(const T& key) const {
  this->separate();
  sync_print("{}: ", thes::json_print(key));
  return {false, this->indent_inner};
}

template<AnyFlushingTag auto tFlush>
template<typename T>
JsonLogger<tFlush, LoggerLevelKind::ARRAY> ObjectLogger<tFlush>::add_array(const T& key) const {
  this->separate();
  sync_print("{}: ", thes::json_print(key));
  return JsonLogger<tFlush, LoggerLevelKind::ARRAY>{this->indent_inner};
}
} // namespace detail

template<AnyFlushingTag auto tFlush>
struct JsonLogger : public detail::JsonLogger<tFlush> {
  using Parent = detail::JsonLogger<tFlush>;

  JsonLogger() : Parent{true} {};
  explicit JsonLogger(decltype(tFlush) /*tag*/) : Parent{true} {};
  explicit JsonLogger(thes::Indentation indent) : Parent{true, indent} {};
  JsonLogger(thes::Indentation indent, decltype(tFlush) /*tag*/) : Parent{true, indent} {};
};
template<AnyFlushingTag TFlush>
JsonLogger(TFlush) -> JsonLogger<TFlush{}>;
template<AnyFlushingTag TFlush>
JsonLogger(thes::Indentation, TFlush) -> JsonLogger<TFlush{}>;
} // namespace lineal

#endif // INCLUDE_LINEAL_ENVIRONMENT_LOGGER_JSON_HPP
