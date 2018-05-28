from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six

import sys
import heapq


def readerLine(fn):
  with open(fn, 'rb') as file:
    while True:
      yield next(file).decode('utf-8').rstrip('\r\n')


def show_top(k, scores, lst):
  h = []
  for i, s in enumerate(scores):
    heapq.heappush(h, (s, -i))
    if len(h) > k:
      heapq.heappop(h)
  num = len(h)
  res = [heapq.heappop(h) for _ in six.moves.range(num)]
  for s, ii in res[::-1]:
    print(' ' + str(s) + '\t' + lst[-ii])


def split_wrt_brackets(str, sp):
  blocks = []
  part = []
  count = 0
  for x in str:
    if count == 0 and x in sp:
      blocks.append(''.join(part))
      part = []
    else:
      part.append(x)
      if x == '(':
        count += 1
      elif x == ')':
        count -= 1
        if count < 0:
          print("Unmatched )", file=sys.stderr)
  blocks.append(''.join(part))
  return blocks
