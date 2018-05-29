#!/usr/bin/env python3
"""
Evaluate GLIMVEC model.
"""
import argparse
from collections import defaultdict
import os
from pathlib import Path
import sys
import logging
import json
import random

import numpy as np
from scipy.stats import rankdata

from ModelKB import Model

logger = logging.getLogger(__name__)


class Evaluator:
    def initialize_ranks(self):
        self.ranks = []

        if self.store_ranking_detail:
            self.ranking_detail = []

    def __init__(self,
                 dataset_path,
                 model_path,
                 *,
                 adjust=False,
                 store_ranking_detail=False,
                 vocab_entity=None,
                 vocab_relation=None):
        if vocab_entity is None:
            vocab_entity = str(dataset_path / 'vocab_entity.txt')
        if vocab_relation is None:
            vocab_relation = str(dataset_path / 'vocab_relation.txt')
        self.e2i = {}
        self.i2e = {}
        for i, line in enumerate(open(vocab_entity)):
            ent = line.split()[0]
            self.e2i[ent] = i
            self.i2e[i] = ent

        self.correct_triples = (
            {
                tuple(line.strip().split())
                for line in (dataset_path / 'train.txt').open()
            }
            | {
                tuple(line.strip().split())
                for line in (dataset_path / 'valid.txt').open()
            }
            | {
                tuple(line.strip().split())
                for line in (dataset_path / 'test.txt').open()
            })

        self.correct_hr2t = defaultdict(list)
        self.correct_tr2h = defaultdict(list)
        for h, r, t in self.correct_triples:
            self.correct_hr2t[(h, r)].append(t)
            self.correct_tr2h[(t, r)].append(h)

        if not adjust:
            self.r2h = json.load((dataset_path / 'most_freq_r2h.json').open())
            self.r2t = json.load((dataset_path / 'most_freq_r2t.json').open())

        self.model = Model(vocab_entity, vocab_relation, str(model_path) + '/')

        self.adjust = adjust
        self.store_ranking_detail = store_ranking_detail

        self.initialize_ranks()

    def correct_tail_entity_indices(self, h, r):
        return [
            self.e2i[t] for t in self.correct_hr2t[(h, r)] if t in self.e2i
        ]

    def correct_head_entity_indices(self, t, r):
        return [
            self.e2i[h] for h in self.correct_tr2h[(t, r)] if h in self.e2i
        ]

    def predict_tail_entities(self, h, r, t):
        scores = self.model.get_score(h, r, True)

        indices = self.correct_tail_entity_indices(h, r)
        if t in self.e2i:
            # remove tail itself
            indices = [i for i in indices if i != self.e2i[t]]

        # do not rank for correct entities
        scores[indices] = -np.inf

        return scores

    def predict_head_entities(self, h, r, t):
        scores = self.model.get_score(t, r, False)

        indices = self.correct_head_entity_indices(t, r)
        if h in self.e2i:
            # remove head itself
            indices = [i for i in indices if i != self.e2i[h]]

        # do not rank for correct entities
        scores[indices] = -np.inf

        return scores

    def get_rank(self, scores, ent):
        if ent in self.e2i:
            return rankdata(-scores, method='min')[self.e2i[ent]]
        else:
            # assume OOV entity's vector is zero
            return (scores > 0.0).sum() + 1

    def get_top10_entities(self, scores):
        top10_indices = np.argpartition(scores, -10)[-10:]
        top10_scores = scores[top10_indices].tolist()
        top10_ents = [self.i2e[i] for i in top10_indices]
        return sorted(
            zip(top10_ents, top10_scores), key=lambda t: t[1], reverse=True)

    def evaluate(self, triples):
        self.initialize_ranks()

        for h, r, t in triples:
            if self.store_ranking_detail:
                instance_info = {}
                instance_info['instance'] = {'h': h, 'r': r, 't': t}

            head_is_oov = h not in self.e2i
            tail_is_oov = t not in self.e2i
            contains_oov_entity = head_is_oov or tail_is_oov

            if contains_oov_entity and self.adjust:
                # just ignore the triple
                continue

            if head_is_oov:
                # pseudo head entity is the most frequent one
                h = self.r2h[r]
            if tail_is_oov:
                # pseudo tail entity is the most frequent one
                t = self.r2t[r]

            scores = self.predict_tail_entities(h, r, t)
            rank = self.get_rank(scores, t)
            self.ranks.append(rank)

            if self.store_ranking_detail:
                instance_info['tail_prediction'] = {
                    'target': {
                        'score': float(scores[self.e2i[t]]),
                        'rank': int(rank),
                    },
                    'top10': [{
                        'ent': ent,
                        'score': float(score),
                    } for ent, score in self.get_top10_entities(scores)]
                }

            scores = self.predict_head_entities(h, r, t)
            rank = self.get_rank(scores, h)
            self.ranks.append(rank)

            if self.store_ranking_detail:
                instance_info['head_prediction'] = {
                    'target': {
                        'score': float(scores[self.e2i[h]]),
                        'rank': int(rank),
                    },
                    'top10': [{
                        'ent': ent,
                        'score': float(score),
                    } for ent, score in self.get_top10_entities(scores)]
                }
                self.ranking_detail.append(instance_info)

    def dump_ranking_detail(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.ranking_detail, f)

    def calc_evaluation_metrics(self):
        ranks = np.array(self.ranks)

        return {
            'MR': ranks.mean(),
            'MRR': (1 / ranks).mean(),
            'Hits@1': 100 * np.count_nonzero(ranks <= 1) / len(ranks),
            'Hits@3': 100 * np.count_nonzero(ranks <= 3) / len(ranks),
            'Hits@10': 100 * np.count_nonzero(ranks <= 10) / len(ranks),
        }


def evaluate(args):
    dataset_path = Path(args.dataset_dir)
    model_path = Path(args.model_dir)
    e = Evaluator(
        dataset_path,
        model_path,
        adjust=args.adjust,
        store_ranking_detail=args.dump_ranking,
        vocab_entity=args.vocab_entity,
        vocab_relation=args.vocab_relation)

    e.evaluate(line.strip().split()
               for line in (dataset_path / '{}.txt'.format(args.split)).open())
    result = e.calc_evaluation_metrics()
    if args.dump_ranking:
        e.dump_ranking_detail(
            str(model_path / 'ranking_detail_{}.json'.format(args.split)))

    print('MR\tMRR\tH@10\tH@3\tH@1')
    print('{MR:.0f}\t{MRR:.3f}\t{Hits@10:.1f}\t{Hits@3:.1f}\t{Hits@1:.1f}'.
          format(**result))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir')
    parser.add_argument('model_dir')
    parser.add_argument(
        '--adjust',
        action='store_true',
        help='Report adjusted scores '
        '(i.e., remove triples with OOV from the test).')
    parser.add_argument('--dump-ranking', action='store_true', default=True)
    parser.add_argument('--vocab-entity')
    parser.add_argument('--vocab-relation')
    parser.add_argument(
        '--split', default='valid', help='test or valid (default)')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.debug(locals())

    evaluate(args)


if __name__ == '__main__':
    main()
