#include "MultinomialTable.h"

#include "RandomGenerator.h"

unsigned int MultinomialTable::sample(RandomGenerator &rd) const {
  size_t i = rd(size);
  unsigned int a = table[i];
  unsigned int b = table[i + 1];
  return (b > a + 1)? a + static_cast<unsigned int>(rd(b - a)) : a;
}
