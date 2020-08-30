import numpy as np
import pandas as pd

import sys
import random
import os

import argparse

import collections

import multiprocessing as mp

import itertools

from tqdm import tqdm

random.seed(420)
np.random.seed(420)

ns = None

def query_worker(seq_id):

    tmp = np.zeros(shape=len(ns.cols)).tolist()
    hits = ns.blast[ns.blast['query_id']==seq_id]
    for i,hit in hits.iterrows():
        idx = ns.lab_pos[hit[1]]*11
        tmp[idx]+=1
        if tmp[idx+10]<hit[11]:
            tmp[idx+1:idx+10]=hit[2:11]
    tmp[-1]=seq_id
    return tmp

# pip install tqdm && cd gen/src && python blast.py --folds 10 --num_threads 94
def main(args):

    global ns

    os.environ['OMP_NUM_THREADS'] = str(args.num_threads)

    labs = pd.read_csv('../data/raw/train_labels.csv').columns[1:]

    cols = []
    lab_pos = dict()
    i = 0
    for lab in labs:
        lab_pos[lab]=i
        i=i+1
        cols.extend([lab+'hits',lab+'identity', lab+'alignment length', lab+'mismatches', lab+'gap opens', lab+'q. start', lab+'q. end', lab+'s. start', lab+'s. end', lab+'evalue', lab+'bit score'])
    cols.extend(['sequence_id'])

    X_train = pd.read_csv('../data/processed/train.csv')
    y_train = pd.read_csv('../data/processed/train.csv')[['target']]
    test = pd.read_csv('../data/processed/test.csv')
    print('Data readed')

    with open('../data/features/blast/train.fasta','w+') as f:
        for i in range(len(X_train)):
            f.write(f'>{i}\n')
            f.write(f'{X_train["sequence"].values[i]}\n')
    print('train fasta file created')

    os.system('makeblastdb -in ../data/features/blast/train.fasta -parse_seqids -title "train_set" -dbtype nucl')
    print('train fasta db created')

    with open('../data/features/blast/query.fasta','w+') as f:
        for i in range(len(test)):
            f.write(f'>{i}\n')
            f.write(f'{test["sequence"].values[i]}\n')
    print('test fasta file created')

    os.makedirs('../data/features/blast/test/',exist_ok=True)

    cutoff = 10
    result_name = '../data/features/blast/test/blast.csv'
    os.system('blastn -db ../data/features/blast/train.fasta -query ../data/features/blast/query.fasta -out '+result_name+' -outfmt 10 -max_target_seqs 9999999 -evalue '+str(cutoff)+' -max_hsps=1 -num_threads '+str(args.num_threads))
    print('query result created')

    os.system('rm ../data/features/blast/*.*')

    mrg = mp.Manager()
    ns = mrg.Namespace()
    ns.cols = cols

    headers = ['query_id', 'subject_id', 'identity', 'alignment length', 'mismatches', 'gap opens', 'q. start', 'q. end', 's. start', 's. end', 'evalue', 'bit score']
    blast = pd.read_csv(result_name, names=headers)
    print('blast df created')

    ids = test.sequence_id.values
    blast.query_id = blast.query_id.apply(lambda x : ids[x])
    blast.subject_id = blast.subject_id.apply(lambda x: y_train.target[x])
    print('Ids and target translated')

    ns.blast = blast
    del blast

    pool = mp.Pool(args.num_threads)

    rows = []
    with tqdm(total=len(ids)) as pbar:
        for i, row in enumerate(pool.imap_unordered(query_worker, ids)):
            rows.append(row)
            pbar.update()

    pool.close()
    pool.join()

    result = pd.DataFrame(rows,columns=cols)
    print('dataset created')
    result_name = '../data/features/blast/test/result.csv'
    print('saving csv')
    result.to_csv(result_name,index=False)
 
    print('PROGRAM END_____')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--folds", default=5,type=int)
    parser.add_argument("--num_threads", default=8,type=int)

    args = parser.parse_args()

    main(args)