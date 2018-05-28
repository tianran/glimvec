# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json
import numpy as np

from utilityFuncs import readerLine


def convert(words, roles, path):
    list_role_pre = [line.split('\t', 1)[0] for line in readerLine(roles)]

    src_list_role = []
    for x in list_role_pre:
        src_list_role.append(x + '<')
        src_list_role.append(x + '>')

    dict_role = dict((s, i) for i, s in enumerate(src_list_role))

    src_mats = np.load(path + 'mats.npy')
    src_msteps = np.load(path + 'msteps.npy')

    tgt_list_role = [x + '>' for x in list_role_pre] + [x + '<' for x in list_role_pre]
    
    tgt_mats = np.empty_like(src_mats)
    tgt_msteps = np.empty_like(src_msteps)
    
    for i, s in enumerate(tgt_list_role):
      tgt_mats[i] = src_mats[dict_role[s]]
      tgt_msteps[i] = src_msteps[dict_role[s]]

    np.save(path + 'mats_conv.npy', tgt_mats)
    np.save(path + 'msteps_conv.npy', tgt_msteps)


if __name__ == '__main__':
    convert(sys.argv[1], sys.argv[2], sys.argv[3])
