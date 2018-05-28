#ifndef GLIMVEC_HYPERPARAMETERSKB_H
#define GLIMVEC_HYPERPARAMETERSKB_H

constexpr unsigned int DIM = 256;
constexpr float SQRT_DIM = 16.0f;

constexpr unsigned int CODE_LEN = 16;
constexpr float SQRT_CLEN = 4.0f;

constexpr float V_ETA = 1.0f / 64;
constexpr float V_LAMBDA = 1.0f / 1024;

constexpr float M_ETA = V_ETA;
constexpr float M_LAMBDA = V_LAMBDA / SQRT_DIM;

constexpr double ORTH_SKIP = 256;
constexpr float ORTH_RATE = 1.0f / 16;
constexpr float ORTH_COEF = 1.0f / 4;

constexpr float AUTOENC_FACTOR = SQRT_DIM * SQRT_CLEN;
constexpr float AUTOENC_ETA = M_ETA * 4 * SQRT_CLEN;
constexpr double AUTOENC_SKIP = 1024;
constexpr float AUTOENC_LAMBDA = M_LAMBDA;
constexpr float JOINT_M_ETA = M_ETA;
constexpr float JOINT_M_LAMBDA = M_LAMBDA / 4;

constexpr bool DISABLE_AUTOENCODER = false;

#endif //GLIMVEC_HYPERPARAMETERSKB_H
