#ifndef __MULTINOMIAL_H
#define __MULTINOMIAL_H


class RandomGenerator;

class Multinomial {
  virtual unsigned int sample(RandomGenerator& rd) const = 0;
};

#endif //__MULTINOMIAL_H
