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

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)

from multiprocessing.reduction import ForkingPickler, AbstractReducer

class ForkingPickler4(ForkingPickler):
    def __init__(self, *args):
        if len(args) > 1:
            args[1] = 2
        else:
            args.append(2)
        super().__init__(*args)

    @classmethod
    def dumps(cls, obj, protocol=4):
        return ForkingPickler.dumps(obj, protocol)


def dump(obj, file, protocol=4):
    ForkingPickler4(file, protocol).dump(obj)


class Pickle4Reducer(AbstractReducer):
    ForkingPickler = ForkingPickler4
    register = ForkingPickler4.register
    dump = dump

ctx = mp.get_context()
ctx.reducer = Pickle4Reducer()

def Worker(f_idx,args):

    print('-------FOLD ',f_idx)
    X_train = pd.read_csv('../data/folds/'+str(args.folds)+'/X_train_split_'+str(f_idx)+'.csv')
    y_train = pd.read_csv('../data/folds/'+str(args.folds)+'/y_train_split_'+str(f_idx)+'.csv')
    test = pd.read_csv('../data/folds/'+str(args.folds)+'/X_dev_split_'+str(f_idx)+'.csv')
    print('-------FOLD ',f_idx,'Data readed')

    with open('../data/features/blast/'+str(f_idx)+'_train.fasta','w+') as f:
        for i in range(len(X_train)):
            f.write(f'>{i}\n')
            f.write(f'{X_train["sequence"].values[i]}\n')
    print('-------FOLD ',f_idx,'train fasta file created')

    os.system('makeblastdb -in ../data/features/blast/'+str(f_idx)+'_train.fasta -parse_seqids -title "train_set" -dbtype nucl')
    print('-------FOLD ',f_idx,'train fasta db created')

    with open('../data/features/blast/'+str(f_idx)+'_query.fasta','w+') as f:
        for i in range(2):
            f.write(f'>{i}\n')
            f.write(f'{test["sequence"].values[i]}\n')
    print('-------FOLD ',f_idx,'test fasta file created')

    os.makedirs('../data/features/blast/'+str(args.folds),exist_ok=True)

    cutoff = 10
    result_name = '../data/features/blast/'+str(args.folds)+"/dev_blast_"+str(f_idx)+".csv"
    os.system('blastn -db ../data/features/blast/'+str(f_idx)+'_train.fasta -query ../data/features/blast/'+str(f_idx)+'_query.fasta -out '+result_name+' -outfmt 10 -max_target_seqs 9999999 -evalue '+str(cutoff)+' -max_hsps=1 -num_threads '+str(args.num_threads))
    print('-------FOLD ',f_idx,'query result created')

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

# pip install tqdm pandarallel && cd gen/src && python blast_folds.py --folds 10 --num_threads 94
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

    jobs = []
    for f_idx in range(1,args.folds+1):
        w = mp.Process(target=Worker, args=(f_idx,args))
        jobs.append(w)
        w.start()

    print('Todos los procesos lanzados')

    for p, job in enumerate(jobs):
        print('esperando proceso ',p+1)
        job.join()
        print('proceso ',p+1,' finalizado')

    os.system('rm ../data/features/blast/*.*')

    mrg = mp.Manager()
    ns = mrg.Namespace()
    ns.cols = cols
    ns.lab_pos = lab_pos
    
    for f_idx in range(1,args.folds+1):

        result_name = '../data/features/blast/'+str(args.folds)+"/dev_blast_"+str(f_idx)+".csv"
        test = pd.read_csv('../data/folds/'+str(args.folds)+'/X_dev_split_'+str(f_idx)+'.csv')
        y_train = pd.read_csv('../data/folds/'+str(args.folds)+'/y_train_split_'+str(f_idx)+'.csv')

        headers = ['query_id', 'subject_id', 'identity', 'alignment length', 'mismatches', 'gap opens', 'q. start', 'q. end', 's. start', 's. end', 'evalue', 'bit score']
        blast = pd.read_csv(result_name, names=headers)
        print('-------FOLD ',f_idx,'blast df created')

        ids = test.sequence_id.values
        blast.query_id = blast.query_id.parallel_apply(lambda x : ids[x])
        blast.subject_id = blast.subject_id.parallel_apply(lambda x: y_train.target[x])
        print('-------FOLD ',f_idx,'Ids and target translated')

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
        result_name = '../data/features/blast/'+str(args.folds)+"/dev_result_"+str(f_idx)+".csv"
        print('saving csv')
        result.to_csv(result_name,index=False)

        print('-------FOLD ',f_idx,'______END')
 
    print('PROGRAM END_____')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--folds", default=5,type=int)
    parser.add_argument("--num_threads", default=8,type=int)

    args = parser.parse_args()

    main(args)