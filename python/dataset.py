#!/usr/bin/env python
import os
from pathlib import Path
import shutil
import tarfile

import click
from logzero import logger
import pandas as pd
import requests


class Dataset:
    _datasets = {
        'kinship': {
            'url':
            'https://github.com/TimDettmers/ConvE/raw/master/kinship.tar.gz',
            'order': ['h', 'r', 't'],
        },
        'nations': {
            'url':
            'https://github.com/TimDettmers/ConvE/raw/master/nations.tar.gz',
            'order': ['h', 'r', 't'],
        },
        'umls': {
            'url':
            'https://github.com/TimDettmers/ConvE/raw/master/umls.tar.gz',
            'order': ['h', 'r', 't'],
        },
        'wn18rr': {
            'url':
            'https://github.com/TimDettmers/ConvE/raw/master/WN18RR.tar.gz',
            'order': ['h', 'r', 't'],
        },
        'fb15k-237': {
            'url':
            'https://github.com/TimDettmers/ConvE/raw/master/FB15k-237.tar.gz',
            'order': ['h', 'r', 't'],
        },
        'wn18': {
            'url':
            'https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:wordnet-mlj12.tar.gz',
            'order': ['h', 'r', 't'],
        },
        'fb15k': {
            'url':
            'https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:fb15k.tgz',
            'order': ['h', 'r', 't'],
        },
    }
    available = list(_datasets.keys())
    _datasets_dir = Path(__file__).parent.parent / 'data'
    _read_table_kwargs = dict(header=None, dtype=str)

    def __init__(self, name, force_download=False):
        name = name.lower()
        if name not in self._datasets:
            raise ValueError('Please check available datasets:',
                             self.available)

        info = self._datasets[name]
        self.name = name
        self.url = info['url']
        self._order_pre = info['order']

        self.data_dir = self._datasets_dir / name

        if not self.data_dir.exists() or force_download:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self._download()
            self._change_order()

        names = ['h', 'r', 't']
        self.train = pd.read_table(
            self.data_dir / 'train.txt',
            names=names,
            **self._read_table_kwargs)
        self.valid = pd.read_table(
            self.data_dir / 'valid.txt',
            names=names,
            **self._read_table_kwargs)
        self.test = pd.read_table(
            self.data_dir / 'test.txt', names=names, **self._read_table_kwargs)
        self.all = pd.concat(
            [self.train, self.valid, self.test], ignore_index=True)
        self.ents = pd.concat([self.all.h, self.all.t]).unique()

        if not (self.data_dir / 'vocab_entity.txt').exists():
            self._make_vocab_file()

        if not (self.data_dir / 'most_freq_r2h.txt').exists():
            self._make_most_freq_r2x_file()

    def _download_wn18(self):
        logger.info('Download %s dataset from:', self.name)
        logger.info(self.url)
        logger.info('Please see the following page for more information:')
        logger.info('https://everest.hds.utc.fr/doku.php?id=en:transe')

        data_tar = self.data_dir / 'wn18.tar.gz'
        r = requests.get(self.url)
        data_tar.write_bytes(r.content)
        tarfile.open(data_tar).extractall(self.data_dir)
        os.remove(data_tar)

        os.rename(self.data_dir / 'wordnet-mlj12' / 'wordnet-mlj12-train.txt',
                  self.data_dir / 'train.txt')
        os.rename(self.data_dir / 'wordnet-mlj12' / 'wordnet-mlj12-valid.txt',
                  self.data_dir / 'valid.txt')
        os.rename(self.data_dir / 'wordnet-mlj12' / 'wordnet-mlj12-test.txt',
                  self.data_dir / 'test.txt')

        os.chmod(self.data_dir / 'valid.txt', 0o664)

        os.remove(self.data_dir / '._wordnet-mlj12')
        shutil.rmtree(self.data_dir / 'wordnet-mlj12')

    def _download_fb15k(self):
        logger.info('Download %s dataset from:', self.name)
        logger.info(self.url)
        logger.info('Please see the following page for more information:')
        logger.info('https://everest.hds.utc.fr/doku.php?id=en:transe')

        data_tar = self.data_dir / 'fb15k.tar.gz'
        r = requests.get(self.url)
        data_tar.write_bytes(r.content)
        tarfile.open(data_tar).extractall(self.data_dir)
        os.remove(data_tar)

        os.rename(self.data_dir / 'FB15k' / 'freebase_mtr100_mte100-train.txt',
                  self.data_dir / 'train.txt')
        os.rename(self.data_dir / 'FB15k' / 'freebase_mtr100_mte100-valid.txt',
                  self.data_dir / 'valid.txt')
        os.rename(self.data_dir / 'FB15k' / 'freebase_mtr100_mte100-test.txt',
                  self.data_dir / 'test.txt')

        os.chmod(self.data_dir / 'train.txt', 0o664)
        os.chmod(self.data_dir / 'valid.txt', 0o664)
        os.chmod(self.data_dir / 'test.txt', 0o664)

        shutil.rmtree(self.data_dir / 'FB15k')

    def _download(self):
        if self.name == 'wn18':
            self._download_wn18()
            return

        if self.name == 'fb15k':
            self._download_fb15k()
            return

        data_tar = self.data_dir / self.url.rsplit('/')[-1]
        logger.info('Download %s dataset from:', self.name)
        logger.info(self.url)
        r = requests.get(self.url)
        data_tar.write_bytes(r.content)
        tarfile.open(data_tar).extractall(self.data_dir)
        os.remove(data_tar)

    def _change_order(self):
        for filename in ['train.txt', 'valid.txt', 'test.txt']:
            file = self.data_dir / filename
            df = pd.read_table(
                file, names=self._order_pre, **self._read_table_kwargs)
            df[['h', 'r', 't']].to_csv(
                file, header=False, index=False, sep='\t')

    def _make_vocab_file(self):
        pd.concat([self.train['h'],
                   self.train['t']]).value_counts().astype('float64').to_csv(
                       self.data_dir / 'vocab_entity.txt',
                       header=False,
                       sep='\t')
        self.train['r'].value_counts().astype('float64').to_csv(
            self.data_dir / 'vocab_relation.txt', header=False, sep='\t')

    def _make_most_freq_r2x_file(self):
        self.train.groupby(['r', 'h']).size().unstack().idxmax(axis=1).to_json(
            self.data_dir / 'most_freq_r2h.json')
        self.train.groupby(['r', 't']).size().unstack().idxmax(axis=1).to_json(
            self.data_dir / 'most_freq_r2t.json')


@click.group()
def cli():
    pass


@cli.command()
@click.option('--dataset', type=click.Choice(Dataset.available))
@click.option('--all', is_flag=True)
@click.option('--debug', is_flag=True)
def download(dataset, all, debug):
    if dataset is not None:
        Dataset(dataset, force_download=not debug)
        return
    if all:
        for dataset in Dataset.available:
            Dataset(dataset, force_download=not debug)
        return


if __name__ == '__main__':
    cli()
