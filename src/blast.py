import numpy as np
import pandas as pd

import sys
import random
import os

import argparse

from tqdm import tqdm

import collections

random.seed(420)
np.random.seed(420)

# pip install tqdm && cd gen/src && python blast.py
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

    for f_idx in tqdm(range(1,args.folds+1)):
        print('-------FOLD ',f_idx)
        X_train = pd.read_csv('../data/folds/'+str(args.folds)+'/X_train_split_'+str(f_idx)+'.csv')
        y_train = pd.read_csv('../data/folds/'+str(args.folds)+'/y_train_split_'+str(f_idx)+'.csv')
        test = pd.read_csv('../data/folds/'+str(args.folds)+'/X_dev_split_'+str(f_idx)+'.csv')
        print('Data readed')

        # make the train sequences into a temporary fasta file with numerical index
        with open('../data/features/blast/train.fasta','w+') as f:
            for i in range(len(X_train)):
                f.write(f'>{i}\n')
                f.write(f'{X_train["sequence"].values[i]}\n')
        print('train fasta file created')

        os.system('makeblastdb -in ../data/features/blast/train.fasta -parse_seqids -title "train_set" -dbtype nucl')
        print('train fasta db created')

        # We will now build a fasta file for our test values, and "query"
        # against the training set sequences
        with open('../data/features/blast/query.fasta','w+') as f:
            for i in range(len(test)):
                f.write(f'>{i}\n')
                f.write(f'{test["sequence"].values[i]}\n')
        print('test fasta file created')

        os.makedirs('../data/features/blast/'+str(args.folds),exist_ok=True)
        # this determines what results are saved. This cutoff is generous
        cutoff = 10
        result_name = '../data/features/blast/'+str(args.folds)+"/dev_blast_"+str(f_idx)+".csv"
        os.system('blastn -db ../data/features/blast/train.fasta -query ../data/features/blast/query.fasta -out '+result_name+' -outfmt 10 -max_target_seqs 9999999 -evalue '+str(cutoff)+' -max_hsps=1 -num_threads '+str(args.num_threads))
        print('query result created')

        headers = ['query_id', 'subject_id', 'identity', 'alignment length', 'mismatches', 'gap opens', 'q. start', 'q. end', 's. start', 's. end', 'evalue', 'bit score']
        blast = pd.read_csv(result_name, names=headers)
        print('blast df created')

        ids = test.sequence_id.values
        blast.query_id = blast.query_id.apply(lambda x : ids[x])
        blast.subject_id = blast.subject_id.apply(lambda x: y_train.target[x])
        print('Ids and target translated')

        result = pd.DataFrame(columns=cols)
        result['sequence_id'] = ids
        for r_idx ,seq_id in tqdm(enumerate(ids),total=len(ids)):
            tmp = np.zeros(shape=len(cols))
            hits = blast[blast['query_id']==seq_id]
            for i,hit in hits.iterrows():
                idx = lab_pos[hit[1]]*11
                tmp[idx]+=1
                if tmp[idx+11]<hit[11]:
                    tmp[idx+1:idx+11]=hit[2:]
            result.iloc[r_idx,0:-1]=tmp
        
        result_name = '../data/features/blast/'+str(args.folds)+"/dev_result_"+str(f_idx)+".csv"
        result.to_csv(result_name,index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--folds", default=5,type=int)
    parser.add_argument("--num_threads", default=10,type=int)

    args = parser.parse_args()

    main(args)