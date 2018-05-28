#include "Poisson.h"

#include <cmath>

#include "RandomGenerator.h"

Poisson::Poisson(double l) : exp_lambda_frac(exp(fmod(l, 512.0))), lambda(l), lambda_left(l), cur(1.0) {}

static bool check_stop(double explf, double rd, double& l, double& c) {
  double ncur = c * rd;
  while (ncur <= 1.0 && l > 0.0) {
    if (l >= 512.0) {
      ncur *= 2.2844135865397565E222; //exp(512.0)
      l -= 512.0;
    } else {
      ncur *= explf;
      l = 0.0;
    }
  }
  return (c = ncur) <= 1.0;
}

unsigned int Poisson::sample(RandomGenerator &rg) const {
  unsigned int ret = 0;
  double l = lambda;
  double c = 1.0;
  while (!check_stop(exp_lambda_frac, rg.nextDouble(), l, c)) ++ret;
  return ret;
}

bool Poisson::stop(RandomGenerator &rg) {
  return check_stop(exp_lambda_frac, rg.nextDouble(), lambda_left, cur);
}

void Poisson::reset() {
  lambda_left = lambda;
  cur = 1.0;
}
