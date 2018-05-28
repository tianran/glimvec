# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from ModelKB import Model
from utilityFuncs import readerLine

def main():
  parser = argparse.ArgumentParser(description='comproleRank.')
  parser.add_argument('words_file', metavar='VOCAB_ENTITY', type=str,
                      help='counts of entities')
  parser.add_argument('roles_file', metavar='VOCAB_RELATION', type=str,
                      help='counts of relations')
  parser.add_argument('model_path', metavar='MODEL_PATH', type=str,
                      help='path to trained model')
  parser.add_argument('comprole_file', metavar='COMPROLE_FILE', type=str,
                      help='compositional constraints')
  args = parser.parse_args()

  model = Model(args.words_file, args.roles_file, args.model_path)

  for line in readerLine(args.comprole_file):
    r1, r2, r, s1, s2 = line.split('\t')
    print(model.mm_rank(r1, r2, r))


if __name__ == '__main__':
  main()
