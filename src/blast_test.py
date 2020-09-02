import numpy as np
import pandas as pd

import os

import argparse

import collections

import multiprocessing as mp

import itertools

from tqdm import tqdm

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=False)


def Query_worker(args):

    hits = args[0][1]
    cols = args[1]
    lab_pos = args[2]

    tmp = np.zeros(shape=len(cols)).tolist()
    seq_id = args[0][0]

    for i,hit in hits.iterrows():
        idx = lab_pos[hit[1]]*11
        tmp[idx]+=1
        if tmp[idx+10]<hit[11]:
            tmp[idx+1:idx+10]=hit[2:11]
    tmp[-1]=seq_id
    return tmp

# cd gen/src && python blast_test.py --num_threads 95 --mode full
def main(args):

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

    print('-------TEST')
    X_train = pd.read_csv('../data/processed/train.csv')
    y_train = X_train.loc[:,'target']
    test = pd.read_csv('../data/processed/test.csv')
    print('-------TEST Data readed')

    result_name = '../data/features/blast/test/blast.csv'

    if ((args.mode == 'full') | (args.mode == 'blast')):

        with open('../data/features/blast/train.fasta','w+') as f:
            for i in range(len(X_train)):
                f.write(f'>{i}\n')
                f.write(f'{X_train["sequence"].values[i]}\n')
        print('-------TEST train fasta file created')

        os.system('makeblastdb -in ../data/features/blast/train.fasta -parse_seqids -title "train_set" -dbtype nucl')
        print('-------TEST train fasta db created')

        with open('../data/features/blast/query.fasta','w+') as f:
            for i in range(len(test)):
                f.write(f'>{i}\n')
                f.write(f'{test["sequence"].values[i]}\n')
        print('-------TEST test fasta file created')

        os.makedirs('../data/features/blast/test/',exist_ok=True)

        cutoff = 10
        os.system('blastn -db ../data/features/blast/train.fasta -query ../data/features/blast/query.fasta -out '+result_name+' -outfmt 10 -max_target_seqs 9999999 -evalue '+str(cutoff)+' -max_hsps=1 -num_threads '+str(args.num_threads))
        print('-------TEST query result created')

        os.system('rm ../data/features/blast/*.*')

    if((args.mode == 'full') | (args.mode == 'query')):

        print('Creating Result DFs...')
        
        headers = ['query_id', 'subject_id', 'identity', 'alignment length', 'mismatches', 'gap opens', 'q. start', 'q. end', 's. start', 's. end', 'evalue', 'bit score']
        blast = pd.read_csv(result_name, names=headers)
        print('-------TEST blast df created')

        ids = test.sequence_id.values
        blast.query_id = blast.query_id.parallel_apply(lambda x : ids[x])
        blast.subject_id = blast.subject_id.parallel_apply(lambda x: y_train[x])
        print('-------TEST Ids and target translated')

        blast = blast.groupby('query_id')

        pool = mp.Pool(args.num_threads)

        rows = []
        with tqdm(total=len(blast)) as pbar:
            for i, row in enumerate(pool.imap_unordered(Query_worker, zip(blast,itertools.repeat(cols),itertools.repeat(lab_pos)))):
                rows.append(row)
                pbar.update()

        pool.close()
        pool.join()

        result = pd.DataFrame(rows,columns=cols)
        print('dataset created')
        result_name = '../data/features/blast/test/blast_result.csv'
        print('saving csv')
        result.to_csv(result_name,index=False)

        print('-------TEST ','______END')
    
    print('PROGRAM END_____')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--num_threads", default=8,type=int)
    parser.add_argument("--mode", default='full',type=str,choices=['full','blast','query'])
    args = parser.parse_args()

    main(args)