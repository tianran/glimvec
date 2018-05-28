#include <Python.h>

#include <thread>
#include <vector>
#include <chrono>
#include <string>
#include <atomic>

#include "RandomGenerator.h"
#include "TrainerKB.h"


class RefPyObj {
  PyObject* obj;

public:
  RefPyObj(): obj() {}
  RefPyObj(PyObject* o): obj(o) {}
  ~RefPyObj() { Py_XDECREF(obj); }
  operator PyObject* () const { return obj; }
  RefPyObj& operator=(PyObject* o) {
    Py_XDECREF(obj);
    obj = o;
    return *this;
  }
  RefPyObj(RefPyObj&& that) noexcept : obj(that.obj) {
    that.obj = nullptr;
  };
  RefPyObj& operator=(RefPyObj&& that) noexcept {
    Py_XDECREF(obj);
    obj = that.obj;
    that.obj = nullptr;
    return *this;
  };
  RefPyObj(const RefPyObj& that) = delete;
  RefPyObj& operator=(const RefPyObj& that) = delete;
};

static RandomGenerator rg(static_cast<uint64_t>(std::chrono::system_clock::now().time_since_epoch().count()));
static std::unique_ptr<TrainerKB> ptrain;

static PyObject* glimvec_initTrainer(PyObject *self, PyObject *args, PyObject *keywds) {
  unsigned int wsz = 0;
  unsigned int rsz = 0;
  const char* inpath = nullptr;
  const char* outpath = nullptr;

  static const char *kwlist[] = {"numEnts", "numRels", "inPath", "outPath", nullptr};

  if (!PyArg_ParseTupleAndKeywords(args, keywds, "ii|zs", (char**)kwlist,
                                   &wsz, &rsz, &inpath, &outpath))
    return nullptr;

  std::string outpathStr;
  if (outpath) outpathStr = std::string(outpath);

  ptrain = std::unique_ptr<TrainerKB>(new TrainerKB());
  ptrain->saveParams(outpathStr);

  if (inpath) ptrain->loadModel(wsz, rsz, inpath);
  else {
    ptrain->initModel(wsz, rsz, rg);
    ptrain->saveModel(outpathStr + "init_");
  }

  Py_RETURN_NONE;
}

static unsigned short glimvec_KB_parseResult(RefPyObj result, unsigned int& hi,
                                             std::vector<std::vector<std::pair<unsigned int, unsigned int>>>& pths) {
  pths.clear();
  if (result && PyTuple_Check(result) && PyTuple_Size(result) == 2) {
    hi = PyLong_AsLong(PyTuple_GetItem(result, 0));
    if (RefPyObj iter_paths = PyObject_GetIter(PyTuple_GetItem(result, 1))) {
      RefPyObj path;
      while ((path = PyIter_Next(iter_paths))) {
        if (RefPyObj iter_edges = PyObject_GetIter(path)) {
          std::vector<std::pair<unsigned int, unsigned int>> pth;
          RefPyObj edge;
          while ((edge = PyIter_Next(iter_edges))) {
            if (PyTuple_Check(edge) && PyTuple_Size(edge) == 2) {
              pth.emplace_back(PyLong_AsLong(PyTuple_GetItem(edge, 0)), PyLong_AsLong(PyTuple_GetItem(edge, 1)));
            } else
              return 4;
          }
          if (!pth.empty()) pths.push_back(std::move(pth));
        } else
          return 3;
      }
    } else
      return 2;
  } else
    return 1;
  return 0;
}

static std::atomic_ushort error;
static const char* err_msg[] = {"a batch should be a (hi, pths) tuple",
                                "pths not iterable",
                                "some pth not iterable",
                                "an edge should be a (ri, ti) tuple",
                                "failed to build tid"};

static std::atomic_llong remained_batches;

static void glimvec_trainKB_para(int tid, RandomGenerator rnd, PyObject* func) {
  PyGILState_STATE gstate;
  gstate = PyGILState_Ensure();

  /* Perform Python actions here. */
  if (RefPyObj arglist = Py_BuildValue("(i)", tid)) {
    while(error.load(std::memory_order_acquire) == 0 && remained_batches.fetch_sub(1, std::memory_order_relaxed) > 0) {
      unsigned int hi;
      std::vector<std::vector<std::pair<unsigned int, unsigned int>>> pths;
      unsigned short msg = glimvec_KB_parseResult(PyObject_CallObject(func, arglist), hi, pths);
      if (msg != 0) {
        error.store(msg, std::memory_order_release);
        break;
      }
      if (pths.empty()) continue;
      Py_BEGIN_ALLOW_THREADS
        ptrain->update(rnd, hi, pths);
      Py_END_ALLOW_THREADS
    }
  } else {
    error.store(5, std::memory_order_release);
  }
  /* Release the thread. No Python API allowed beyond this point. */
  PyGILState_Release(gstate);
}

static PyObject* glimvec_trainKB(PyObject *self, PyObject *args, PyObject *keywds) {
  PyObject* func = nullptr;
  long long numBatches = 100000;
  int para = 2;

  static const char *kwlist[] = {"batchGenFunc", "numBatches", "para", nullptr};

  if (!PyArg_ParseTupleAndKeywords(args, keywds, "O|Li", (char**)kwlist,
                                   &func, &numBatches, &para))
    return nullptr;

  if (!PyCallable_Check(func)) {
    PyErr_SetString(PyExc_TypeError, "parameter must be callable");
    return nullptr;
  }
  Py_INCREF(func);
  Py_BEGIN_ALLOW_THREADS
    error.store(0, std::memory_order_release);
    remained_batches.store(numBatches, std::memory_order_release);
    std::vector<std::thread> threads;
    threads.reserve(para);
    for (int i = 0; i != para; ++i) {
      rg.jump();
      threads.emplace_back(&glimvec_trainKB_para, i, rg, func);
    }
    for (auto& x : threads) x.join();
  Py_END_ALLOW_THREADS
  Py_DECREF(func);

  unsigned short err = error.load(std::memory_order_acquire);
  if (err != 0) {
    PyErr_SetString(PyExc_ValueError, err_msg[err - 1]);
    return nullptr;
  }

  Py_RETURN_NONE;
}

static PyObject* glimvec_saveModel(PyObject *self, PyObject *args, PyObject *keywds) {
  const char* outpath = nullptr;

  static const char *kwlist[] = {"outPath", nullptr};

  if (!PyArg_ParseTupleAndKeywords(args, keywds, "|s", (char**)kwlist,
                                   &outpath))
    return nullptr;

  std::string outpathStr;
  if (outpath) outpathStr = std::string(outpath);

  ptrain->saveModel(outpathStr);

  Py_RETURN_NONE;
}

static PyMethodDef GlimvecMethods[] = {
    {"initTrainer",  (PyCFunction)glimvec_initTrainer, METH_VARARGS | METH_KEYWORDS, "Init Trainer."},
    {"trainKB",  (PyCFunction)glimvec_trainKB, METH_VARARGS | METH_KEYWORDS, "Train Model from Knowledge Base."},
    {"saveModel",  (PyCFunction)glimvec_saveModel, METH_VARARGS | METH_KEYWORDS, "Save Model."},
    {nullptr, nullptr, 0, nullptr}        /* Sentinel */
};

static struct PyModuleDef glimvec = {
    PyModuleDef_HEAD_INIT,
    "glimvec",   /* name of module */
    nullptr,  /* module documentation, may be NULL */
    0,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    GlimvecMethods
};

PyMODINIT_FUNC PyInit_glimvec() {
  return PyModule_Create(&glimvec);
}