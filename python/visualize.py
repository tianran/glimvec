#!/usr/bin/env python3
"""
Visualize relation matrices.
"""
import argparse
from collections import defaultdict, Counter
from pathlib import Path
import os
import sys
import logging
from itertools import chain

from bokeh.palettes import Category20
import click
import numpy as np
from scipy.stats import rankdata
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm
import logzero
from logzero import logger
from bokeh.plotting import figure, save, output_file
from bokeh.models import ColumnDataSource, LinearColorMapper, HoverTool, LabelSet, CategoricalColorMapper
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from umap import UMAP

from ModelKB import Model


@click.command()
@click.argument(
    'dataset_dir',
    type=click.Path(exists=True, file_okay=False, resolve_path=True))
@click.argument(
    'model_dir',
    type=click.Path(exists=True, file_okay=False, resolve_path=True))
@click.option(
    '--out-file',
    type=click.Path(),
    help='Path to saved plot (Default: MODEL_DIR/plot.html).')
@click.option('--text-label', is_flag=True)
@click.option(
    '--method',
    type=click.Choice(['umap', 't-sne']),
    default='umap',
    help='Manifold learning algorithm (Default: umap).')
@click.option(
    '--n-neighbors',
    default=15,
    help=
    'The number of neighbors to use to approximate geodesic distance. Used only for UMAP.'
)
@click.option(
    '--metric',
    default='euclidean',
    help=
    'The metric to use to compute distances in high dimensional space. Used only for UMAP.'
)
@click.option(
    '--coloring',
    type=click.Choice(['coding', 'step', 'coding-argmax']),
    default='coding',
    help='Default: coding')
@click.option('--random-state', default=810)
@click.option(
    '--vocab-entity', type=click.Path(exists=True, resolve_path=True))
@click.option(
    '--vocab-relation', type=click.Path(exists=True, resolve_path=True))
@click.option('-v', '--verbose', count=True)
def visualize(dataset_dir, model_dir, out_file, text_label, method,
              n_neighbors, metric, coloring, random_state, vocab_entity,
              vocab_relation, verbose):
    logzero.loglevel(logging.DEBUG if verbose > 0 else logging.INFO)
    logger.debug(locals())

    out_file = (Path(model_dir) /
                'plot.html').resolve() if out_file is None else out_file

    dataset_path = Path(dataset_dir)
    vocab_entity = (dataset_path / 'vocab_entity.txt'
                    ).resolve() if vocab_entity is None else vocab_entity
    vocab_relation = (dataset_path / 'vocab_relation.txt'
                      ).resolve() if vocab_relation is None else vocab_relation
    model = Model(vocab_entity, vocab_relation, model_dir + '/')

    rels = model.list_role
    vecs = [model.mats[model.dict_role[r]].flatten() for r in rels]
    X = normalize(np.vstack(vecs), axis=1)

    logger.debug('data shape: %s', X.shape)
    logger.info('calculating {} embeddings...'.format(method))

    if method == 't-sne':
        X_emb = TSNE(
            n_components=2, random_state=random_state).fit_transform(X)
    elif method == 'umap':
        X_emb = UMAP(
            n_components=2,
            random_state=random_state,
            metric=metric,
            n_neighbors=n_neighbors).fit_transform(X)
    else:
        raise NotImplementedError

    logger.info('...done')

    df = pd.DataFrame(X_emb, columns=['x', 'y'], index=rels)
    df['label_tail'] = df.index.str.split('/').map(lambda l: l[-1])
    df['label_full'] = df.index
    df['step'] = [model.msteps[model.dict_role[r]] for r in rels]
    df['step_log'] = np.log(df['step'])

    codes = np.vstack(model.code_of(role) for role in rels)

    if coloring == 'step':
        color_mapper = {
            'field':
            'step_log',
            'transform':
            LinearColorMapper(
                palette='Inferno256',
                low=df['step_log'].min(),
                high=df['step_log'].max())
        }
        additional_tooltips = [('mstep', '@step')]
    elif coloring == 'coding-argmax':
        df['code_argmax'] = [str(i) for i in codes.argmax(axis=1)]
        color_mapper = {
            'field':
            'code_argmax',
            'transform':
            CategoricalColorMapper(
                factors=list(str(i) for i in range(codes.shape[1])),
                palette=Category20[codes.shape[1]])
        }
        additional_tooltips = []
    elif coloring == 'coding':

        def to_rgb(h):
            h = h.lstrip('#')
            return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

        def to_hex(r, g, b):
            return '#{}{}{}'.format(
                hex(int(r))[2:], hex(int(g))[2:], hex(int(b))[2:])

        np.random.seed(random_state)
        colors_r_init = np.random.normal(size=codes.shape[1])
        colors_g_init = np.random.normal(size=codes.shape[1])
        colors_b_init = np.random.normal(size=codes.shape[1])
        codes_norm = codes / codes.sum(axis=1).reshape(-1, 1)
        colors_norm = np.column_stack(((codes_norm * colors_r_init).sum(
            axis=1), (codes_norm * colors_g_init).sum(axis=1),
                                       (codes_norm * colors_b_init).sum(
                                           axis=1)))
        colors = minmax_scale(colors_norm, (0, 255)).astype(int)
        df['code_color'] = [to_hex(r, g, b) for r, g, b in colors]

        color_mapper = 'code_color'

        df['r'] = colors[:, 0]
        df['g'] = colors[:, 1]
        df['b'] = colors[:, 2]
        additional_tooltips = [('(r, g, b)', '(@r, @g, @b)')]
    else:
        raise ValueError

    output_file(out_file)

    p = figure(plot_width=1000, plot_height=1000)
    source = ColumnDataSource(df)
    p.scatter(x='x', y='y', size=14, source=source, color=color_mapper)
    hover = HoverTool(tooltips=[
        ('index', '$index'),
        ('(x,y)', '($x, $y)'),
        ('relation', '@label_full'),
        *additional_tooltips,
    ])
    p.add_tools(hover)

    if text_label:
        labels = LabelSet(
            x='x',
            y='y',
            text='label_tail',
            level='glyph',
            x_offset=5,
            y_offset=5,
            source=source)
        p.add_layout(labels)

    save(p)
    logger.info('save plot to %s', out_file)


if __name__ == '__main__':
    visualize()
