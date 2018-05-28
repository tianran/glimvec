#include <iostream>
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <chrono>
#include <unordered_map>
#include <cmath>
#include <utility>

#include "optparse.h"
#include "ReaderLines.h"
#include "RandomGenerator.h"
#include "Poisson.h"
#include "TrainerKB.h"
#include "MultinomialTable.h"
#include "misc.h"

using namespace std;
using namespace misc;

class option : public optparse {
public:
  bool help = false;

  double sampPow = 0.75;
  double sampPathLen = 0.5;
  long long numBatches = 1000000;
  const char* inPath = nullptr;
  string outPath;
  int para = 2;

  BEGIN_OPTION_MAP()
    ON_OPTION(SHORTOPT('h') || LONGOPT("help"))
      help = true;
    ON_OPTION_WITH_ARG(LONGOPT("sampPow"))
      sampPow = stod(arg);
    ON_OPTION_WITH_ARG(LONGOPT("sampPathLen"))
      sampPathLen = stod(arg);
    ON_OPTION_WITH_ARG(LONGOPT("numBatches"))
      numBatches = stoll(arg);
    ON_OPTION_WITH_ARG(LONGOPT("inPath"))
      inPath = arg;
    ON_OPTION_WITH_ARG(LONGOPT("outPath"))
      outPath = string(arg);
    ON_OPTION_WITH_ARG(LONGOPT("para"))
      para = stoi(arg);

  END_OPTION_MAP()
};

static vector<vector<pair<unsigned int, unsigned int>>> graph; // neighbors: (relation_index, tail_index)
static MultinomialTable samp_node;
static atomic_ullong remained_batches;

static void trainKB_para(int tid, RandomGenerator rnd, double pl, TrainerKB* ptrain) {
  Poisson samp_path(pl);

  long long remained;
  while((remained = remained_batches.fetch_sub(1, memory_order_relaxed)) > 0) {
    if (remained % 100000 == 0) {
      cerr << remained << endl;
    }
    unsigned int hi = samp_node.sample(rnd);

    vector<vector<pair<unsigned int, unsigned int>>> pths;
    unsigned int samp_sz = 0;
    const auto& neighbor = graph[hi];
    for (unsigned int i = 0; i != neighbor.size() * 2; ++i) {
      vector<pair<unsigned int, unsigned int>> pth;
      auto edge = neighbor[rnd(neighbor.size())];
      samp_path.reset();
      do {
        pth.push_back(edge);
        if (++samp_sz == 31) break;
        const auto& nei = graph[edge.second];
        edge = nei[rnd(nei.size())];
      } while (!samp_path.stop(rnd));
      pths.push_back(move(pth));
      if (samp_sz == 31) break;
    }
    ptrain->update(rnd, hi, pths);
  }
}

int main(int argc, char *argv[])
{
  try {
    option opt;
    int argpos = opt.parse(argv, argc);
    if (opt.help) {
      cout << "Train model for KB." << endl
           << "  trainKB [OPTION...] VOCAB_ENTITY VOCAB_RELATION TRAIN_FILE" << endl
           << endl << "positional arguments:" << endl
           << "  VOCAB_ENTITY      counts of entities" << endl
           << "  VOCAB_RELATION    counts of relations" << endl
           << "  TRAIN_FILE        train file" << endl
           << endl << "optional arguments:" << endl
           << "  -h, --help        show this help message and exit" << endl
           << "  --sampPow         samp. node prob. is power of freq. (default: 0.75)" << endl
           << "  --sampPathLen     path length is 1+Poisson(sampPathLen) (default: 0.5)" << endl
           << "  --numBatches      batches to train (default: 1000000)" << endl
           << "  --inPath          if set, load model from this path for init" << endl
           << "  --outPath         save model to this path (default: working dir)" << endl
           << "  --para            number of parallel threads (default: 2)" << endl
          ;
      return 0;
    }
    if (argc - argpos != 3) throw runtime_error("wrong number of arguments");
    string words_fn(argv[argpos]);
    string roles_fn(argv[argpos + 1]);
    string train_fn(argv[argpos + 2]);

    // read vocab of entities
    unordered_map<string, unsigned int> words;
    unsigned int wsz = 0; {
      vector<double> wprobs;
      ReaderLines wlines(words_fn);
      while (!wlines.empty()) {
        auto sp = split(wlines.next(), '\t');
        words[sp[0]] = wsz;
        ++wsz;
        wprobs.push_back(stod(sp[1]));
      }
      for (auto& x : wprobs) x = pow(x, opt.sampPow);
      samp_node = MultinomialTable(wprobs.cbegin(), wprobs.cend(), 1 << 16);
    }

    // read vocab of relations
    unordered_map<string, unsigned int> roles;
    unsigned int rsz = 0; {
      ReaderLines rlines(roles_fn);
      while (!rlines.empty()) {
        roles[split(rlines.next(), '\t')[0]] = rsz;
        ++rsz;
      }
    }

    //read train file, add neighbors to graph
    graph.resize(wsz);
    ReaderLines glines(train_fn);
    while (!glines.empty()) {
      auto sp = split(glines.next(), '\t');
      const unsigned int head_index = words.at(sp[0]);
      const unsigned int tail_index = words.at(sp[2]);
      const unsigned int rel_index = roles.at(sp[1]);
      graph[head_index].emplace_back(rel_index, tail_index);
      graph[tail_index].emplace_back(rel_index + rsz, head_index);
    }

    RandomGenerator rg(static_cast<uint64_t>(chrono::system_clock::now().time_since_epoch().count()));

    TrainerKB trainer;
    trainer.saveParams(opt.outPath);
    if (opt.inPath) trainer.loadModel(wsz, rsz, opt.inPath);
    else {
      trainer.initModel(wsz, rsz, rg);
      trainer.saveModel(opt.outPath + "init_");
    }

    vector<thread> threads;
    threads.reserve(opt.para);

    remained_batches = opt.numBatches;
    for (int i = 0; i != opt.para; ++i) {
      rg.jump();
      threads.emplace_back(&trainKB_para, i, rg, opt.sampPathLen, &trainer);
    }
    for (auto& x : threads) x.join();

    trainer.saveModel(opt.outPath);

  } catch (const optparse::unrecognized_option& e) {
    cout << "unrecognized option: " << e.what() << endl;
    return 1;
  } catch (const optparse::invalid_value& e) {
    cout << "invalid value: " << e.what() << endl;
    return 1;
  } catch (const exception& e) {
    cout << "use -h or --help to show help." << endl;
    cout << e.what() << endl;
  }

  return 0;
}
