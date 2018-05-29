# GLIMVEC

GLIMVEC (Graph to LInear Mappings and VECtors) is a tool for constructing embeddings for knowledge graphs.

Paper: [Interpretable and Compositional Relation Learning by Joint Training with an Autoencoder](https://arxiv.org/abs/1805.09547)

## Prerequisites

* General: Python3
* For compiling the trainer from source: c++, make
    * For re-compiling the Python module: CPython(>=3.5)
    * For compiling in Windows: [Visual Studio Build Tools](http://landinghub.visualstudio.com/visual-cpp-build-tools)

## Detailed Performance on Benchmark Datasets

TBW

## Usage for Evaluation:

To reproduce results in the ACL2018 [paper](https://arxiv.org/abs/1805.09547):

    $ for dataset in {wn18,fb15k,wn18rr,fb15k-237}; do echo ${dataset}; for setting in {joint,base,jointcomp,basecomp}; do echo " model-${setting}"; python python/evaluate.py --split test data/${dataset} acl2018/${dataset}/model-${setting}; done; done

## Usage for Training:

You can either use a python module, or compile a stand alone executable for training.

### Use the pre-built python module:

We have included pre-built binary python modules for several operating systems. If the versions feel right, this is the easiest way to start training.

* For Linux: (compiled in Ubuntu 16.04 LTS, with Python 3.5 and gcc 5.4)

    `$ cp pre-built/linux/glimvec.so python`

* For MacOS: (compiled in High Sierra, with Python 3.6 and Apple LLVM 9.1)

    `$ cp pre-built/macos/glimvec.so python`

* For Windows: (compiled in Windows 10, using Python 3.6 in Anaconda 5.1 and Build Tools for Visual Studio 2017)

    `$ copy pre-built\windows\glimvec.pyd python`

Then, run the python training script and show help:

    $ python python/trainKB.py -h
    usage: trainKB.py [-h] [--sampPow SAMPPOW] [--sampPathLen SAMPPATHLEN]
              [--numBatches NUMBATCHES] [--inPath INPATH]
              [--outPath OUTPATH] [--para PARA]
              [--glimvecModule GLIMVECMODULE]
              VOCAB_ENTITY VOCAB_RELATION TRAIN_FILE

    Train model for KB.

    positional arguments:
      VOCAB_ENTITY          counts of entities
      VOCAB_RELATION        counts of relations
      TRAIN_FILE            train file

    optional arguments:
      -h, --help            show this help message and exit
      --sampPow SAMPPOW     sampling nodes by probabilities proportional to the
                power of frequency. (default: 0.75)
      --sampPathLen SAMPPATHLEN
                path length is 1+Poisson(sampPathLen) (default: 0.5)
      --numBatches NUMBATCHES
                batches to train (default: 1000000)
      --inPath INPATH       if set, load model from this path for init
      --outPath OUTPATH     save model to this path (default: working dir)
      --para PARA           number of parallel threads (default: 2)
      --glimvecModule GLIMVECMODULE
                path to the pre-trained python library (default: None)

Example for training on the `nations` dataset:

    $ mkdir -p model/nations
    $ python python/trainKB.py --numBatches 1000 --outPath model/nations/ data/nations/vocab_entity.txt data/nations/vocab_relation.txt data/nations/train.txt

Trained model is stored under `model/nations` directory:

    $ ls model/nations
    cvecs.npy  decoder.npy  dstep.npy  encoder.npy  mats.npy  msteps.npy  params.json  tvecs.npy  vsteps.npy
              
### Compile a stand alone training executable from source:

You can also compile a stand alone executable for training. The training speed will be about 1.3 times faster than the python module.

First, get the Eigen library:

    $ cd build
    $ git clone https://github.com/eigenteam/eigen-git-mirror.git
    $ cd eigen-git-mirror
    $ git checkout branches/3.3
    $ cd ../..

Then, depending on your OS, run the following:

* In Linux:

    `$ cp cpp/Makefile.linux build/Makefile`

* In MacOS:

    `$ cp cpp/Makefile.macos build/Makefile`

* In Windows:

    `$ copy cpp\Makefile.windows build\Makefile`

    If you use Build Tools for Visual Studio 2017, run the following to set up a C++ compilation environment:

    `$ C:\"Program Files (x86)"\"Microsoft Visual Studio"\2017\BuildTools\VC\Auxiliary\Build\vcvarsall.bat x64`

Now compile:

    $ cd build
    $ make
    $ cd ..

This will produce an executable `trainKB`. Run the following to show help:

    $ build/trainKB --help
    Train model for KB.
      trainKB [OPTION...] VOCAB_ENTITY VOCAB_RELATION TRAIN_FILE

    positional arguments:
      VOCAB_ENTITY      counts of entities
      VOCAB_RELATION    counts of relations
      TRAIN_FILE        train file

    optional arguments:
      -h, --help        show this help message and exit
      --sampPow         samp. node prob. is power of freq. (default: 0.75)
      --sampPathLen     path length is 1+Poisson(sampPathLen) (default: 0.5)
      --numBatches      batches to train (default: 1000000)
      --inPath          if set, load model from this path for init
      --outPath         save model to this path (default: working dir)
      --para            number of parallel threads (default: 2)

Example for training on the `nations` dataset:

    $ mkdir -p model/nations
    $ build/trainKB --numBatches 1000 --outPath model/nations/ data/nations/vocab_entity.txt data/nations/vocab_relation.txt data/nations/train.txt

### Re-compile the Python module:

If the pre-built python modules do not work, and you have succeeded in compiling a stand alone executable but still want to use Python, try the following to re-compile the Python module:

* In Linux and MacOS:

    `$ cd build; make glimvec.so; cd ..`

* In Windows:

    `$ cd build; make glimvec.pyd; cd ..`

You may want to change `PYTHON3_LIB`, `PYTHON3_LIB_PATH` and `PYTHON3_INCLUDE` in the Makefile for successful compiling.
