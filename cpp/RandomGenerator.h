#ifndef __RANDOMGENERATOR_H
#define __RANDOMGENERATOR_H

#include <cstdint>
#include <string>

/* xoroshiro128+ generator initialized by SplitMix64, jump at init */
class RandomGenerator {

  uint64_t s[2];
  uint64_t rotl(const uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

public:

  typedef uint64_t result_type;

  static constexpr uint64_t min() { return 0ULL; }
  static constexpr uint64_t max() { return ~0ULL; }

  explicit RandomGenerator(uint64_t seed);
  uint64_t operator()();

  uint64_t operator()(uint64_t len) { return operator()() % len; }

  double nextDouble() {
    constexpr uint64_t DOUBLE_MASK = (1ULL << 53) - 1;
    constexpr double NORM_53 = 1.0 / (1ULL << 53);
    return (operator()() & DOUBLE_MASK) * NORM_53;
  }

  float nextFloat() {
    constexpr uint64_t FLOAT_MASK = (1 << 24) - 1;
    constexpr float NORM_24 = 1.0f / (1 << 24);
    return (operator()() & FLOAT_MASK) * NORM_24;
  }

  void jump();

  std::string toString();
};


#endif //__RANDOMGENERATOR_H
