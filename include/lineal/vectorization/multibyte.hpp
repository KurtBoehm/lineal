// This file is part of https://github.com/KurtBoehm/lineal.
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef INCLUDE_LINEAL_VECTORIZATION_MULTIBYTE_HPP
#define INCLUDE_LINEAL_VECTORIZATION_MULTIBYTE_HPP

#include <array>
#include <cstddef>

#include <immintrin.h>

#include "thesauros/macropolis.hpp"
#include "thesauros/static-ranges.hpp"
#include "thesauros/types.hpp"

#include "lineal/vectorization/grex.hpp"

namespace grex {
template<typename TContainer, std::size_t tCount, std::size_t tPart = tCount>
inline constexpr auto shuffle_mask =
  thes::star::index_transform<tCount * TContainer::int_bytes>([](auto i) {
    constexpr std::size_t int_byte_num = TContainer::int_bytes;
    constexpr std::size_t elem_byte_num = TContainer::element_bytes;
    constexpr std::size_t elems_per_chunk = sizeof(__m128i) / int_byte_num;

    constexpr std::size_t j = i % int_byte_num;
    constexpr std::size_t k = i / int_byte_num;
    constexpr std::size_t l = k % elems_per_chunk;
    return (j < elem_byte_num && k < tPart) ? std::byte{thes::u8(j + elem_byte_num * l)}
                                            : std::byte{128};
  }) |
  thes::star::to_array;

template<typename TContainer, std::size_t tCount>
inline constexpr auto shuffle_masks = thes::star::index_transform<tCount + 1>([](auto ci) {
                                        return shuffle_mask<TContainer, tCount, ci>;
                                      }) |
                                      thes::star::to_array;

namespace multibyte_detail {
inline constexpr std::size_t bytes128 = 16;
inline constexpr std::size_t bytes256 = 32;
inline constexpr std::size_t bytes512 = 64;

template<std::size_t tByteNum>
struct VectorLoader {};

template<>
struct VectorLoader<bytes128> {
  static __m128i load(const std::byte* ptr) {
    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
  }
};

#if STADO_INSTRUCTION_SET >= STADO_AVX2
template<>
struct VectorLoader<bytes256> {
  static __m256i load(const std::byte* ptr) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
  }
};
#endif // AVX2

#if STADO_INSTRUCTION_SET >= STADO_AVX512F
template<>
struct VectorLoader<bytes512> {
  static __m512i load(const std::byte* ptr) {
    return _mm512_loadu_si512(ptr);
  }
};
#endif // AVX512
} // namespace multibyte_detail

inline __m128i load128i(const std::byte* ptr) {
  return multibyte_detail::VectorLoader<multibyte_detail::bytes128>::load(ptr);
}

#if STADO_INSTRUCTION_SET >= STADO_AVX2
inline __m256i load256i(const std::byte* ptr) {
  return multibyte_detail::VectorLoader<multibyte_detail::bytes256>::load(ptr);
}
#endif // AVX2

#if STADO_INSTRUCTION_SET >= STADO_AVX512F
inline __m512i load512i(const std::byte* ptr) {
  return multibyte_detail::VectorLoader<multibyte_detail::bytes512>::load(ptr);
}
#endif // AVX512

namespace multibyte_detail {
template<typename TContainer>
inline constexpr std::size_t block_ints = bytes128 / TContainer::int_bytes;
template<typename TContainer>
inline constexpr std::size_t block_jump = TContainer::element_bytes * block_ints<TContainer>;

template<typename TIt, std::size_t tCount>
struct MultiByteShuffler;

// 128 bit

template<typename TIt, std::size_t tCount>
requires(TIt::Container::int_bytes* tCount <= bytes128)
struct MultiByteShuffler<TIt, tCount> {
  static __m128i zero() {
    return _mm_setzero_si128();
  }

  template<std::size_t tPart>
  static __m128i load(const TIt& it, const std::array<std::byte, bytes128>& shuffle_array) {
    const __m128i raw_data = load128i(it.raw());
    const __m128i shuffler = load128i(shuffle_array.data());
    return _mm_shuffle_epi8(raw_data, shuffler);
  }
};

// 256 bit

#if STADO_INSTRUCTION_SET >= STADO_AVX2
template<typename TIt, std::size_t tCount>
requires(TIt::Container::int_bytes* tCount == bytes256)
struct MultiByteShuffler<TIt, tCount> {
  using Container = TIt::Container;

  static __m256i zero() {
    return _mm256_setzero_si256();
  }

  template<std::size_t tPart>
  static __m256i load(const TIt& it, const std::array<std::byte, bytes256>& shuffle_array) {
    const std::byte* ptr = it.raw();
    const __m128i raw_lower = load128i(ptr);
    const __m256i raw = [ptr, raw_lower]() THES_ALWAYS_INLINE {
      if constexpr (tPart > block_ints<Container>) {
        const __m128i raw_data2 = load128i(ptr + block_jump<Container>);
        return _mm256_setr_m128i(raw_lower, raw_data2);
      } else {
        return _mm256_castsi128_si256(raw_lower);
      }
    }();
    const __m256i shuffler = load256i(shuffle_array.data());
    return _mm256_shuffle_epi8(raw, shuffler);
  }
};
#endif // AVX2

// 512 bit

#if STADO_INSTRUCTION_SET >= STADO_AVX512F
template<typename TIt, std::size_t tCount>
requires(TIt::Container::int_bytes* tCount == bytes512)
struct MultiByteShuffler<TIt, tCount> {
  using Container = TIt::Container;

  static __m512i zero() {
    return _mm512_setzero_si512();
  }

  template<std::size_t tPart>
  static __m512i load(const TIt& it, const std::array<std::byte, bytes512>& shuffle_array) {
    static constexpr std::size_t jump = block_jump<Container>;
    static constexpr std::size_t ints = block_ints<Container>;

    const std::byte* ptr = it.raw();
    __m512i raw = _mm512_castsi128_si512(load128i(ptr));
    if constexpr (tPart > ints) {
      raw = _mm512_inserti32x4(raw, load128i(ptr + jump), 1);
    }
    if constexpr (tPart > 2 * ints) {
      raw = _mm512_inserti32x4(raw, load128i(ptr + 2 * jump), 2);
    }
    if constexpr (tPart > 3 * ints) {
      raw = _mm512_inserti32x4(raw, load128i(ptr + 3 * jump), 3);
    }
    const __m512i shuffler = load512i(shuffle_array.data());
    return _mm512_shuffle_epi8(raw, shuffler);
  }
};
#endif // AVX512

template<typename TVector>
struct RegisterToVector {
  static TVector transform(auto v) {
    return v;
  }
};
template<typename T, std::size_t tSize>
struct RegisterToVector<stado::SubNativeVector<T, tSize>> {
  using Vector = stado::SubNativeVector<T, tSize>;
  static Vector transform(auto v) {
    return Vector::from_register(v);
  }
};
} // namespace multibyte_detail

template<typename TIter, std::size_t tSize>
struct MultiByteLoader {
  using Container = TIter::Container;
  using Value = Container::Value;
  using Vector = grex::Vector<Value, tSize>;
  using Shuffler = multibyte_detail::MultiByteShuffler<TIter, tSize>;
  static constexpr std::size_t register_bytes = sizeof(typename Vector::Register);
  static constexpr std::size_t element_bytes = Container::element_bytes;
  static constexpr std::size_t int_bytes = Container::int_bytes;
  static constexpr std::size_t padding_bytes = Container::padding_bytes;
  static constexpr std::size_t shuffle_num = register_bytes / sizeof(Value);
  static_assert(padding_bytes >= register_bytes);

  template<std::size_t tPart>
  static Vector load_part(const TIter& it) {
    if constexpr (tPart == 0) {
      return Shuffler::zero();
    }
    if constexpr (int_bytes == element_bytes) {
      return no_transform(it).cutoff(tPart);
    }
    return to_vector(
      Shuffler::template load<tPart>(it, shuffle_mask<Container, shuffle_num, tPart>));
  }

  static Vector load_part(const TIter& it, std::size_t part) {
    if constexpr (int_bytes == element_bytes) {
      return no_transform(it).cutoff(part);
    }
    return to_vector(
      Shuffler::template load<tSize>(it, shuffle_masks<Container, shuffle_num>[part]));
  }

  static Vector load(const TIter& it) {
    if constexpr (int_bytes == element_bytes) {
      return no_transform(it);
    }
    return to_vector(Shuffler::template load<tSize>(it, shuffle_mask<Container, shuffle_num>));
  }

private:
  static Vector to_vector(const auto reg) {
    return multibyte_detail::RegisterToVector<Vector>::transform(reg);
  }
  static Vector no_transform(const TIter& it) {
    return to_vector(multibyte_detail::VectorLoader<register_bytes>::load(it.raw()));
  }
};

template<typename TIter>
struct MultiByteLoader<TIter, 1> {
  using Container = TIter::Container;
  using Integer = Container::Value;
  using Vector = stado::SingleVector<Integer>;

  template<std::size_t tPart>
  static Vector load_part(const TIter& it) {
    if constexpr (tPart > 0) {
      return Vector(*it);
    } else {
      return Vector(0);
    }
  }

  static Vector load_part(const TIter& it, std::size_t part) {
    if (part > 0) {
      return Vector(*it);
    }
    return Vector(0);
  }

  static Vector load(const TIter& it) {
    return Vector(*it);
  }
};

template<typename TIt, std::size_t tCount>
inline auto load_multibyte(const TIt& it, VectorSize<tCount> /*vec_size*/) {
  return MultiByteLoader<TIt, tCount>::load(it);
}
template<typename TIt, std::size_t tCount>
inline auto load_multibyte(const TIt& it, VectorPartSize<tCount> vec_size) {
  return MultiByteLoader<TIt, tCount>::load_part(it, vec_size.part());
}
} // namespace grex

#endif // INCLUDE_LINEAL_VECTORIZATION_MULTIBYTE_HPP
