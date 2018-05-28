#ifndef __MULTINOMIALTABLE_H
#define __MULTINOMIALTABLE_H

#include "Multinomial.h"

#include <vector>
#include <cstddef>
#include <memory>
#include <utility>


class MultinomialTable : public Multinomial {

  size_t size;
  std::unique_ptr<unsigned int[]> table;
  std::vector<double> scan;

public:
  MultinomialTable() = default;

  template <typename InputIter>
  MultinomialTable(InputIter prob_begin, InputIter prob_end, size_t sz) :
      size(sz), table(new unsigned int[sz + 1]) {

    double total = 0.0;
    for (InputIter cur = prob_begin; cur != prob_end; ++cur) {
      total += *cur;
      scan.push_back(total);
    }

    size_t lower = 0;
    auto scan_sz = static_cast<unsigned int>(scan.size());
    for (unsigned int i = 0; i != scan_sz; ++i) {
      auto higher = static_cast<size_t>((scan[i] /= total) * sz);
      while (lower <= higher) table[lower++] = i;
    }
    table[sz] = scan_sz;
  }

  double prob(unsigned int i) const { return scan[i]; }
  unsigned int choices() const { return static_cast<unsigned int>(scan.size()); }

  unsigned int sample(RandomGenerator& rd) const override;
};


#endif //__MULTINOMIALTABLE_H
