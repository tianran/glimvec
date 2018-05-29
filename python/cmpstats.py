#!/usr/bin/env python3
import argparse
from itertools import chain
import json
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', nargs='+')
    parser.add_argument(
        '--split', default='valid', help='test or valid (default)')
    args = parser.parse_args()

    if len(args.model_dir) < 1:
        logger.error('require at least one model')
        sys.exit(1)
    for md in args.model_dir:
        if not (Path(md) /
                'ranking_detail_{}.json'.format(args.split)).exists():
            logger.error('%s/ranking_detail_%s.json not found.', md,
                         args.split)
            logger.error('need to run evaluate.py for %s', md)
            sys.exit(1)

    def gather_ranks_from_ranking_json(filepath):
        return list(
            chain.from_iterable(((instance['head_prediction']['target'][
                'rank'], instance['tail_prediction']['target']['rank'])
                                 for instance in json.load(open(filepath)))))

    def calc_evaluation_metrics(ranks):
        ranks = np.array(ranks)
        return {
            'MR': ranks.mean(),
            'MRR': (1 / ranks).mean(),
            'Hits@1': 100 * np.count_nonzero(ranks <= 1) / len(ranks),
            'Hits@3': 100 * np.count_nonzero(ranks <= 3) / len(ranks),
            'Hits@10': 100 * np.count_nonzero(ranks <= 10) / len(ranks),
        }

    metrics = []
    for md in args.model_dir:
        md = Path(md)
        ranks = gather_ranks_from_ranking_json(
            str(md / 'ranking_detail_{}.json'.format(args.split)))
        metrics.append(calc_evaluation_metrics(ranks))

    columns = ['MR', 'MRR', 'Hits@10', 'Hits@3', 'Hits@1']
    stats = pd.DataFrame(
        metrics, columns=columns).describe().loc[['mean', 'std']]
    print(stats)


if __name__ == '__main__':
    main()
