# -*- coding: utf-8 -*-

import argparse
import math
import numpy as np
import importlib
import importlib.util

from utilityFuncs import readerLine

def main():
  parser = argparse.ArgumentParser(description='Train model for KB.')
  parser.add_argument('words_file', metavar='VOCAB_ENTITY', type=str,
                      help='counts of entities')
  parser.add_argument('roles_file', metavar='VOCAB_RELATION', type=str,
                      help='counts of relations')
  parser.add_argument('train_file', metavar='TRAIN_FILE', type=str,
                      help='train file')

  parser.add_argument('--sampPow', dest='sampPow', type=float, default=0.75,
                      help='sampling nodes by probabilities proportional to the power of frequency. (default: 0.75)')
  parser.add_argument('--sampPathLen', dest='sampPathLen', type=float, default=0.5,
                      help='path length is 1+Poisson(sampPathLen) (default: 0.5)')
  parser.add_argument('--numBatches', dest='numBatches', type=int, default=1000000,
                      help='batches to train (default: 1000000)')
  parser.add_argument('--inPath', dest='inPath', type=str, default=None,
                      help='if set, load model from this path for init')
  parser.add_argument('--outPath', dest='outPath', type=str, default="",
                      help='save model to this path (default: working dir)')
  parser.add_argument('--para', dest='para', type=int, default=2,
                      help='number of parallel threads (default: 2)')
  parser.add_argument('--glimvecModule', dest='glimvecModule', type=str, default=None,
                      help='path to the pre-trained python library (default: None)')

  args = parser.parse_args()

  # read vocab of entities
  words = {}
  wsz = 0
  wprobs = []
  for line in readerLine(args.words_file):
    w, freq = line.split('\t')
    words[w] = wsz
    wsz += 1
    wprobs.append(math.pow(float(freq), args.sampPow))
  samp_node_prob = np.array(wprobs) / np.sum(wprobs)

  # read vocab of relations
  roles = {}
  rsz = 0
  for line in readerLine(args.roles_file):
    roles[line.split('\t')[0]] = rsz
    rsz += 1

  # read train file, add neighbors to graph
  graph = [[]] * wsz
  for line in readerLine(args.train_file):
    head, rel, tail = line.split('\t')
    head_index = words[head]
    rel_index = roles[rel]
    tail_index = words[tail]
    graph[head_index].append((rel_index, tail_index))
    graph[tail_index].append((rel_index + rsz, head_index))

  # function for generating batch
  #  sample size in a batch should not exceed 31
  def genBatch(tid):
    hi = np.random.choice(wsz, p=samp_node_prob)
    pths = []
    samp_sz = 0
    neighbor = graph[hi]
    for i in range(len(neighbor) * 2):
      pth = []
      edge = neighbor[np.random.choice(len(neighbor))]
      pth.append(edge)
      samp_sz += 1
      if samp_sz < 31:
        for _ in range(np.random.poisson(args.sampPathLen)):
          nei = graph[edge[1]]
          edge = nei[np.random.choice(len(nei))]
          pth.append(edge)
          samp_sz += 1
          if samp_sz == 31:
            break
      pths.append(pth)
      if samp_sz == 31:
        break
    return hi, pths

  # to debug, first check if the genBatch function is ok:
  #print(genBatch(0))

  if args.glimvecModule is None:
    glimvec = importlib.import_module('glimvec')
  else:
    spec = importlib.util.spec_from_file_location("glimvec", args.glimvecModule)
    glimvec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(glimvec)

  glimvec.initTrainer(wsz, rsz, inPath=args.inPath, outPath=args.outPath)
  glimvec.trainKB(genBatch, args.numBatches, args.para)
  glimvec.saveModel(args.outPath)


if __name__ == '__main__':
  main()
