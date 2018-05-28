#ifndef GLIMVEC_TRAINERKB_H
#define GLIMVEC_TRAINERKB_H

#include <vector>
#include <string>
#include <memory>
#include <atomic>

#include "Eigen/Core"

#include "RandomGenerator.h"
#include "Poisson.h"

class TrainerKB {

  Eigen::MatrixXf ctvecs;
  std::vector<Eigen::MatrixXf> mats;
  Eigen::MatrixXf encoder;
  Eigen::MatrixXf decoder;

  std::unique_ptr<std::atomic_ullong[]> v_steps;
  std::unique_ptr<std::atomic_ullong[]> m_steps;
  std::atomic_ullong denc_step;

  float sigtab[1537];

  void mincr_regularize(unsigned int mi, RandomGenerator& rnd);

public:
  TrainerKB();

  void saveParams(const std::string& outPath);

  void update(RandomGenerator& rnd, unsigned int hi,
              const std::vector<std::vector<std::pair<unsigned int, unsigned int>>>& pths);

  void saveModel(const std::string& outPath);
  void loadModel(unsigned int wsz, unsigned int rsz, const std::string& inPath);
  void initModel(unsigned int wsz, unsigned int rsz, RandomGenerator& rg);
};


#endif //GLIMVEC_TRAINERKB_H
