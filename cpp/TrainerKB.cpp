#include "TrainerKB.h"

#include <fstream>
#include <cmath>
#include <random>

#include "HyperParametersKB.h"
#include "misc.h"

using namespace std;
using namespace Eigen;
using namespace misc;


/* hyper parameters. */
static constexpr float vEta = V_ETA;
static constexpr float mEta = M_ETA;
static constexpr double orthSkip = ORTH_SKIP;
static constexpr float orthRate = ORTH_RATE;
static constexpr float orthEL = (orthRate / orthSkip) * (M_LAMBDA * SQRT_DIM / ORTH_COEF);
static constexpr float autoFactor = AUTOENC_FACTOR;
static constexpr float autoEta = AUTOENC_ETA;
static constexpr double autoSkip = AUTOENC_SKIP;
static constexpr float jointMEta = JOINT_M_ETA;
static constexpr float jointM_EL = JOINT_M_ETA * JOINT_M_LAMBDA;

static constexpr float vEL = V_ETA * V_LAMBDA;
static constexpr float mEL = M_ETA * M_LAMBDA;
static constexpr float autoEL = AUTOENC_ETA * AUTOENC_LAMBDA;

static constexpr bool disableAutoencoder = DISABLE_AUTOENCODER;

TrainerKB::TrainerKB() {
  for (unsigned int i = 0; i != 256 * 6; ++i)
    sigtab[i] = static_cast<float>(1.0 / (exp(i / 256.0) + 1.0) - 0.5);
  sigtab[256 * 6] = -0.5f;

  Eigen::initParallel();
}
/* for experiments, one can slightly change the code to dynamically pass
 * hyper-parameters through constructor, instead of hard coding. */

void TrainerKB::saveParams(const string &outPath) {
  ofstream out_params(outPath + "params.json");
  out_params.precision(17);
  out_params << scientific << '{' << endl
             << "  \"trainer\" : \"TrainerKB\"," << endl
             << "  \"dim\" : " << DIM << ',' << endl
             << "  \"codeLen\" : " << CODE_LEN << ',' << endl
             << "  \"vEta\" : " << vEta << ',' << endl
             << "  \"mEta\" : " << mEta << ',' << endl
             << "  \"orthSkip\" : " << orthSkip << ',' << endl
             << "  \"orthRate\" : " << orthRate << ',' << endl
             << "  \"orthEL\" : " << orthEL << ',' << endl
             << "  \"autoFactor\" : " << autoFactor << ',' << endl
             << "  \"autoEta\" : " << autoEta << ',' << endl
             << "  \"autoSkip\" : " << autoSkip << ',' << endl
             << "  \"jointMEta\" : " << jointMEta << ',' << endl
             << "  \"jointM_EL\" : " << jointM_EL << ',' << endl
             << "  \"vEL\" : " << vEL << ',' << endl
             << "  \"mEL\" : " << mEL << ',' << endl
             << "  \"autoEL\" : " << autoEL << ',' << endl
             << "  \"disableAutoencoder\" : " << disableAutoencoder << endl
             << '}' << endl;
  out_params.close();
}

static string array_string(const Ref<const ArrayXf>& a) {
  return mkString(a.data(), a.data() + a.size(), "[", ", ", "]\n");
}

static string vec_string(const Ref<const VectorXf>& v) {
  return mkString(v.data(), v.data() + 8, "[", ", ", "...]\n");
}

void TrainerKB::update(RandomGenerator &rnd, unsigned int hi,
                       const vector<vector<pair<unsigned int, unsigned int>>> &pths) {

  MatrixXf twv(DIM, 128);
  MatrixXf unwv(DIM, 256);

  unsigned int tdest[128];
  unsigned int unis[128];

  unsigned int inter_tvi[32];
  unsigned int inter_mi[32];
  float inter_mnrm[32];

  unsigned int samp_sz = 0;
  hi += ctvecs.cols() / 2;
  twv.col(0) = (1.0f / (vEL * static_cast<float>(v_steps[hi].load(memory_order_relaxed)) + 1.0f)) * ctvecs.col(hi);
  unsigned int csz = 1;

  for (const auto& pth : pths) {
    vector<unsigned int> calcs;
    calcs.reserve(pth.size() + 1);
    calcs.push_back(0);
    for (unsigned int pth_index = 0; pth_index != pth.size(); ++pth_index) {
      const unsigned int samp_sz4 = samp_sz * 4;
      const unsigned int un_index = samp_sz4 + 128;
      {
        const unsigned int ui = pth[pth_index].second;
        unwv.col(un_index) = (1.0f / (vEL * static_cast<float>(v_steps[ui].load(memory_order_relaxed)) + 1.0f)) * ctvecs.col(ui);
        unis[samp_sz] = ui;
      }
      unsigned int choice = rnd(calcs.size());
      inter_tvi[samp_sz] = calcs[choice];
      for (unsigned int j = pth_index; j != choice; --j) {
        MatrixXf m = mats[pth[j].first];
        unwv.col(un_index) = sqrtf(DIM / m.squaredNorm()) * (m * unwv.col(un_index));

        debug_print("unwv@%d: mi = %d\n", un_index, pth[j].first);
      }{
        MatrixXf m = mats[pth[pth_index].first];
        twv.col(csz) = sqrtf(DIM / m.squaredNorm()) * (m.transpose() * twv.col(calcs.back()));
        calcs.push_back(csz);
        tdest[samp_sz] = csz++;

        debug_print("twv: mi = %d, src = %d, dest = %d\n", pth[pth_index].first, calcs.back(), csz - 1);
      }{
        const unsigned int calcs_choice1 = calcs[choice + 1];
        for (unsigned int k = 1; k != 4; ++k) {
          const unsigned int samp_sz_k32 = samp_sz + k * 32;
          vector<unsigned int> nmis(pth_index - choice); {
            const unsigned int un_index_k = un_index + k;
            const unsigned int ni = rnd(ctvecs.cols() / 2);
            unwv.col(un_index_k) = (1.0f / (vEL * static_cast<float>(v_steps[ni].load(memory_order_relaxed)) + 1.0f)) * ctvecs.col(ni);
            unis[samp_sz_k32] = ni;
            for (auto& x : nmis) {
              x = rnd(mats.size());
              MatrixXf m = mats[x];
              unwv.col(un_index_k) = sqrtf(DIM / m.squaredNorm()) * (m * unwv.col(un_index_k));

              debug_print("unwv@%d: mi = %d\n", un_index_k, x);
            }
          }
          if (nmis.empty()) {
            tdest[samp_sz_k32] = calcs_choice1;
          } else {
            tdest[samp_sz_k32] = samp_sz_k32;
            auto rev = nmis.crbegin(); {
              MatrixXf m = mats[*rev];
              twv.col(samp_sz_k32) = sqrtf(DIM / m.squaredNorm()) * (m.transpose() * twv.col(calcs_choice1));

              debug_print("twv: mi = %d, src = %d, dest = %d\n", *rev, calcs_choice1, samp_sz_k32);
            }
            for (++rev; rev != nmis.crend(); ++rev) {
              MatrixXf m = mats[*rev];
              twv.col(samp_sz_k32) = sqrtf(DIM / m.squaredNorm()) * (m.transpose() * twv.col(samp_sz_k32));

              debug_print("twv: mi = %d, src = %d, dest = %d\n", *rev, samp_sz_k32, samp_sz_k32);
            }
          }
        }
      }
      const unsigned int mi = pth[choice].first;
      inter_mi[samp_sz] = mi;
      MatrixXf mat = mats[mi];
      float nrm = sqrtf(mat.squaredNorm() / DIM);
      inter_mnrm[samp_sz] = fminf(nrm / (mEL * static_cast<float>(m_steps[mi].load(memory_order_relaxed)) + 1.0f), 4.0f);
      unwv.middleCols(samp_sz4, 4) = (1.0f / nrm) * (mat * unwv.middleCols(un_index, 4));

      debug_print("unwv4-128@%d: mi = %d\n", samp_sz4, pth[choice].first);

      while (choice-- != 0) {
        MatrixXf m = mats[pth[choice].first];
        unwv.middleCols(samp_sz4, 4) = sqrtf(DIM / m.squaredNorm()) * (m * unwv.middleCols(samp_sz4, 4));

        debug_print("unwv4@%d: mi = %d\n", samp_sz4, pth[choice].first);
      }
      ++samp_sz;
    }
  }

  const unsigned int samp_sz4 = samp_sz * 4;
  ArrayXf dots = (unwv.leftCols(samp_sz4).transpose() * twv.col(0)).array() * 256.0f - 281.24475f;
  ArrayXf sigs = (dots.abs() + 0.5f).min(1536.0f);
  dots = dots.sign();
  Map<ArrayXf, 0, InnerStride<4>>(dots.data(), samp_sz) =
                      -Map<ArrayXf, 0, InnerStride<4>>(dots.data(), samp_sz);
  for (unsigned int k = 0; k != samp_sz4; ++k) sigs(k) = sigtab[static_cast<int>(sigs(k))];
  sigs = sigs * dots - 0.5f;
  Map<ArrayXf, 0, InnerStride<4>>(sigs.data(), samp_sz) =
                      -Map<ArrayXf, 0, InnerStride<4>>(sigs.data(), samp_sz);

  debug_print("dots = %s\n", array_string((unwv.leftCols(samp_sz4).transpose() * twv.col(0)).array()).c_str());
  debug_print("sigs = %s\n", array_string(sigs).c_str());

  for (unsigned int k = 0; k != samp_sz; ++k) {
    for (unsigned int l = 0; l != 4; ++l) {
      const unsigned int idx = k + l * 32;
      const unsigned int des = tdest[idx];
      const unsigned int uni = unis[idx];
      ctvecs.col(uni) += vEta * 8.0f / fmaxf(twv.col(des).norm(), 8.0f) * sigs(k * 4 + l) * twv.col(des);
      v_steps[uni].fetch_add(1, memory_order_relaxed);

      debug_print("t_norm[%d] = %e\n", idx, twv.col(des).squaredNorm());
      debug_print("unv@%d += %s\n", uni, vec_string(vEta * 8.0f / fmaxf(twv.col(des).norm(), 8.0f) * sigs(k * 4 + l) * twv.col(des)).c_str());
    }
  }
  ArrayXf un_norm = vEta * 8.0f * sigs / unwv.leftCols(samp_sz4).colwise().norm().transpose().array().max(8.0f);
  ctvecs.col(hi) += unwv.leftCols(samp_sz4) * un_norm.matrix();
  v_steps[hi].fetch_add(samp_sz4, memory_order_relaxed);

  debug_print("un_norm = %s\n", array_string(unwv.leftCols(samp_sz4).colwise().squaredNorm().array()).c_str());
  debug_print("tv@%d += %s\n", hi, vec_string(unwv.leftCols(samp_sz4) * un_norm.matrix()).c_str());

  for (unsigned int k = 0; k != samp_sz; ++k) {
    const unsigned int mi = inter_mi[k];
    const unsigned int tvi = inter_tvi[k];
    mats[mi] += twv.col(tvi) * (unwv.middleCols(128 + k * 4, 4) *
                                (mEta * 64.0 * inter_mnrm[k] / fmaxf(twv.col(tvi).norm(), 8.0f) * sigs.segment(k * 4, 4) /
                                 unwv.middleCols(128 + k * 4, 4).colwise().norm().transpose().array().max(8.0f)).matrix()).transpose();
    mincr_regularize(mi, rnd);

    debug_print("inter_tnrm[%d] = %e\n", k, twv.col(tvi).squaredNorm());
    debug_print("un_norm[%d ~ %d] = %s\n", 128 + k * 4, 128 + k * 4 + 3, array_string(unwv.middleCols(128 + k * 4, 4).colwise().squaredNorm().transpose().array()).c_str());
    debug_print("M@%d: tv = %s\n", inter_mi[k], vec_string(mEta * 8.0f * inter_mnrm[k] / fmaxf(twv.col(tvi).norm(), 8.0f) * twv.col(tvi)).c_str());
    debug_print("M@%d: unv = %s\n", inter_mi[k], vec_string(unwv.middleCols(128 + k * 4, 4) * (8.0f * sigs.segment(k * 4, 4) / unwv.middleCols(128 + k * 4, 4).colwise().norm().transpose().array().max(8.0f)).matrix()).c_str());
  }

  debug_print("update\n");
}

static string denc_string(const Ref<const MatrixXf>& m) {
  string ret;
  for (unsigned int l = 0; l != 4; ++l) {
    for (unsigned int k = 0; k != 8; ++k) {
      ret += mkString(m.data() + DIM * DIM * l + DIM * k, m.data() + DIM * DIM * l + DIM * k + 8, "[", ", ", "...]\n");
    }
    ret += "...\n\n";
  }
  return ret;
}

void TrainerKB::mincr_regularize(unsigned int mi, RandomGenerator& rnd) {
  const unsigned long long mstep = m_steps[mi].fetch_add(1, memory_order_relaxed) + 1;
  float mscal = 1.0f / (mEL * static_cast<float>(mstep) + 1.0f);
  if (!disableAutoencoder && rnd.nextDouble() * autoSkip < 1.0) {
    const unsigned long long dstep = denc_step.fetch_add(1, memory_order_relaxed);
    const float denc_scal = 1.0f / (autoEL * static_cast<float>(dstep) + 1.0f);

    const unsigned int ni1 = rnd(mats.size());
    const unsigned int ni2 = rnd(mats.size());
    const unsigned int ni3 = rnd(mats.size());
    MatrixXf mni_copy(DIM * DIM, 4);
    mni_copy.col(0) = Map<VectorXf>(mats[mi].data(), DIM * DIM);
    mni_copy.col(1) = Map<VectorXf>(mats[ni1].data(), DIM * DIM);
    mni_copy.col(2) = Map<VectorXf>(mats[ni2].data(), DIM * DIM);
    mni_copy.col(3) = Map<VectorXf>(mats[ni3].data(), DIM * DIM);

    ArrayX4f codes = (encoder.transpose() * mni_copy).array();
    Array<float, 1, 4> reci_norms = SQRT_DIM / mni_copy.colwise().norm().array();
    codes.rowwise() *= denc_scal * reci_norms;
    codes = codes.min(4.0f * SQRT_DIM);
    ArrayX4f codes_hinge = (0.5f + 0.25f * codes).max(0.0f);
    ArrayX4f codes_grad = codes_hinge.min(1.0f);
    ArrayX4f crelus = codes_grad * (2.0f * codes_hinge).max(codes);

    debug_print("code_relu = %s\n", array_string(crelus.col(0)).c_str());

    MatrixXf outs = decoder * crelus.matrix();

    debug_print("outs_norms = %s\n", array_string(outs.colwise().squaredNorm().transpose().array()).c_str());

    Array4f dots = (256.0f / AUTOENC_FACTOR) * denc_scal * reci_norms(0) * (outs.transpose() * mni_copy.col(0)).array() - 281.24475f;

    debug_print("mdots = %s\n", array_string(denc_scal * reci_norms(0) * (outs.transpose() * mni_copy.col(0)).array()).c_str());

    Array4f sigs = (dots.abs() + 0.5f).min(1536.0f);
    dots = dots.sign();
    dots(0) = -dots(0);
    for (unsigned int k = 0; k != 4; ++k) sigs(k) = sigtab[static_cast<int>(sigs(k))];
    sigs = sigs * dots - 0.5f;
    sigs(0) = -sigs(0);

    const float rate = (jointMEta / AUTOENC_FACTOR) * fminf(mscal / reci_norms(0), 4.0f) /
        ((jointM_EL * static_cast<float>(mstep) / autoSkip + 1.0f) * mscal);
    Map<VectorXf>(mats[mi].data(), DIM * DIM) +=
        outs * (rate * sigs * ((16.0f * DIM * CODE_LEN) / outs.colwise().squaredNorm().transpose().array()).sqrt().min(denc_scal)).matrix();

    sigs *= autoEta / AUTOENC_FACTOR;

    encoder += mni_copy * (((denc_scal * reci_norms(0) * (decoder.transpose() * mni_copy.col(0))).array().max(-4.0f * SQRT_DIM).min(4.0f * SQRT_DIM).matrix() *
        (sigs.matrix().transpose().array() * reci_norms).matrix()).array() * codes_grad).matrix().transpose();

    debug_print("encoder += \n%s\n", denc_string(mni_copy * (((denc_scal * reci_norms(0) * (decoder.transpose() * mni_copy.col(0))).array().max(-4.0f * SQRT_DIM).min(4.0f * SQRT_DIM).matrix() *
                                                              (sigs.matrix().transpose().array() * reci_norms).matrix()).array() * codes_grad).matrix().transpose()).c_str());

    decoder += mni_copy.col(0) * (crelus.matrix() * (reci_norms(0) * sigs).matrix()).transpose();

    debug_print("decoder += \n%s\n", denc_string(mni_copy.col(0) * (crelus.matrix() * (reci_norms(0) * sigs).matrix()).transpose()).c_str());
  }
  if (rnd.nextDouble() * orthSkip < 1.0) {
    const MatrixXf ma = mats[mi];
    MatrixXf m2 = ma * ma.transpose();
    const float ma_nrm = m2.trace() / DIM;
    m2 -= ma_nrm * MatrixXf::Identity(DIM, DIM);
    const float rate = -orthRate / ma_nrm * fminf(mscal, 4.0f / sqrtf(ma_nrm)) /
        ((orthEL * static_cast<float>(mstep) / orthSkip + 1.0f) * mscal);
    mats[mi] += rate * m2 * ma;
  }
}

void TrainerKB::saveModel(const string &outPath) {
  const void *data;
  union {
    unsigned long long l;
    char c[sizeof(unsigned long long)];
  } ulc;
  {
    const unsigned int wsz2 = ctvecs.cols();
    const unsigned int wsz = wsz2 / 2;
    const string vecs_header = createNpyHeader<float>(false, {wsz, DIM});

    ofstream out_cvecs(outPath + "cvecs.npy");
    out_cvecs << vecs_header;
    data = ctvecs.data();
    out_cvecs.write(static_cast<const char *>(data), DIM * wsz * sizeof(float));
    out_cvecs.close();
    ofstream out_tvecs(outPath + "tvecs.npy");
    out_tvecs << vecs_header;
    data = ctvecs.data() + DIM * wsz;
    out_tvecs.write(static_cast<const char *>(data), DIM * wsz * sizeof(float));
    out_tvecs.close();
    ofstream out_vsteps(outPath + "vsteps.npy");
    out_vsteps << createNpyHeader<unsigned long long>(false, {wsz2});
    for (unsigned int i = 0; i != wsz2; ++i) {
      ulc.l = v_steps[i];
      out_vsteps.write(ulc.c, sizeof(unsigned long long));
    }
    out_vsteps.close();
  }{
    const unsigned int rsz2 = mats.size();

    ofstream out_mats(outPath + "mats.npy");
    out_mats << createNpyHeader<float>(false, {rsz2, DIM, DIM});;
    for (const auto& m : mats) {
      data = m.data();
      out_mats.write(static_cast<const char *>(data), DIM * DIM * sizeof(float));
    }
    out_mats.close();

    ofstream out_msteps(outPath + "msteps.npy");
    out_msteps << createNpyHeader<unsigned long long>(false, {rsz2});
    for (unsigned int i = 0; i != rsz2; ++i) {
      ulc.l = m_steps[i];
      out_msteps.write(ulc.c, sizeof(unsigned long long));
    }
    out_msteps.close();
  }
  string denc_header = createNpyHeader<float>(false, {CODE_LEN, DIM, DIM});
  ofstream out_encoder(outPath + "encoder.npy");
  out_encoder << denc_header;
  data = encoder.data();
  out_encoder.write(static_cast<const char *>(data), DIM * DIM * CODE_LEN * sizeof(float));
  out_encoder.close();
  ofstream out_decoder(outPath + "decoder.npy");
  out_decoder << denc_header;
  data = decoder.data();
  out_decoder.write(static_cast<const char *>(data), DIM * DIM * CODE_LEN * sizeof(float));
  out_decoder.close();
  ofstream out_dstep(outPath + "dstep.npy");
  out_dstep << createNpyHeader<unsigned long long>(false, {});
  ulc.l = denc_step;
  out_dstep.write(ulc.c, sizeof(unsigned long long));
  out_dstep.close();

  debug_print("saveModel Done.\n");
}

void TrainerKB::loadModel(unsigned int wsz, unsigned int rsz, const string &inPath) {
  void* data;
  union {
    char c[sizeof(unsigned long long)];
    unsigned long long l;
  } ucl;
  {
    const unsigned int wsz2 = wsz * 2;
    ctvecs.resize(DIM, wsz2);
    ifstream in_cvecs(inPath + "cvecs.npy");
    checkNpyHeader<float>(in_cvecs, {wsz, DIM});
    data = ctvecs.data();
    in_cvecs.read(static_cast<char *>(data), DIM * wsz * sizeof(float));
    in_cvecs.close();
    ifstream in_tvecs(inPath + "tvecs.npy");
    checkNpyHeader<float>(in_tvecs, {wsz, DIM});
    data = ctvecs.data() + DIM * wsz;
    in_tvecs.read(static_cast<char *>(data), DIM * wsz * sizeof(float));
    in_tvecs.close();

    v_steps = unique_ptr<atomic_ullong[]>(new atomic_ullong[wsz2]);
    ifstream in_vsteps(inPath + "vsteps.npy");
    checkNpyHeader<unsigned long long>(in_vsteps, {wsz2});
    for (unsigned int i = 0; i != wsz2; ++i) {
      in_vsteps.read(ucl.c, sizeof(unsigned long long));
      v_steps[i] = ucl.l;
    }
    in_vsteps.close();
  }{
    const unsigned int rsz2 = rsz * 2;
    mats.resize(rsz2);

    ifstream in_mats(inPath + "mats.npy");
    checkNpyHeader<float>(in_mats, {rsz2, DIM, DIM});
    for (auto& m : mats) {
      m.resize(DIM, DIM);
      data = m.data();
      in_mats.read(static_cast<char *>(data), DIM * DIM * sizeof(float));
    }
    in_mats.close();

    ifstream in_msteps(inPath + "msteps.npy");
    m_steps = unique_ptr<atomic_ullong[]>(new atomic_ullong[rsz2]);
    checkNpyHeader<unsigned long long>(in_msteps, {rsz2});
    for (unsigned int i = 0; i != rsz2; ++i) {
      in_msteps.read(ucl.c, sizeof(unsigned long long));
      m_steps[i] = ucl.l;
    }
    in_msteps.close();
  }
  encoder.resize(DIM * DIM, CODE_LEN);
  ifstream in_encoder(inPath + "encoder.npy");
  checkNpyHeader<float>(in_encoder, {CODE_LEN, DIM, DIM});
  data = encoder.data();
  in_encoder.read(static_cast<char *>(data), DIM * DIM * CODE_LEN * sizeof(float));
  in_encoder.close();
  decoder.resize(DIM * DIM, CODE_LEN);
  ifstream in_decoder(inPath + "decoder.npy");
  checkNpyHeader<float>(in_decoder, {CODE_LEN, DIM, DIM});
  data = decoder.data();
  in_decoder.read(static_cast<char *>(data), DIM * DIM * CODE_LEN * sizeof(float));
  in_decoder.close();
  ifstream in_dstep(inPath + "dstep.npy");
  checkNpyHeader<unsigned long long>(in_dstep, {});
  in_dstep.read(ucl.c, sizeof(unsigned long long));
  denc_step = ucl.l;
  in_dstep.close();

  debug_print("loadModel Done.\n");
}

void TrainerKB::initModel(unsigned int wsz, unsigned int rsz, RandomGenerator &rg) {
  debug_print("wsz: %d, rsz: %d\n", wsz, rsz);
  debug_print("%s\n", rg.toString().c_str());

  normal_distribution<float> gaus(0.0f, static_cast<float>(1.0 / sqrt(DIM)));
  {
    const unsigned int wsz2 = wsz * 2;
    ctvecs.resize(DIM, wsz2);
    for (float *p = ctvecs.data(); p != ctvecs.data() + DIM * wsz; ++p) *p = gaus(rg);
    ctvecs.rightCols(wsz) = ctvecs.leftCols(wsz);
    v_steps = unique_ptr<atomic_ullong[]>(new atomic_ullong[wsz2]);
    for (unsigned int i = 0; i != wsz2; ++i) v_steps[i] = 0;
  }{
    const unsigned int rsz2 = rsz * 2;
    mats.resize(rsz * 2);
    for (auto& m : mats) {
      m.resize(DIM, DIM);
      for (unsigned int i = 0; i != DIM; ++i) {
        for (unsigned int j = 0; j != DIM; ++j) {
          float tmp = gaus(rg) * 0.5f;
          if (i == j) tmp += 0.5f;
          m(j, i) = tmp;
        }
      }
    }
    m_steps = unique_ptr<atomic_ullong[]>(new atomic_ullong[rsz2]);
    for (unsigned int i = 0; i != rsz2; ++i) m_steps[i] = 0;
  }
  encoder.resize(DIM * DIM, CODE_LEN);
  for (float *p = encoder.data(); p != encoder.data() + DIM * DIM * CODE_LEN; ++p) *p = gaus(rg);
  decoder = encoder;
  denc_step = 0;

  debug_print("%s\n", rg.toString().c_str());
  debug_print("initModel Done.\n");
}
