#ifndef __POISSON_H
#define __POISSON_H


class RandomGenerator;

class Poisson {

  const double exp_lambda_frac;
  const double lambda;

  double lambda_left;
  double cur;

public:
  explicit Poisson(double l);
  unsigned int sample(RandomGenerator& rg) const;

  bool stop(RandomGenerator& rg);
  void reset();
};


#endif //__POISSON_H
