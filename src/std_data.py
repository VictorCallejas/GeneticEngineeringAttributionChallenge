import pandas as pd 
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold
from tqdm.notebook import tqdm

from sklearn.preprocessing import RobustScaler

from transformers import RobertaTokenizerFast

data_path = '../data/'

def main():

    train = pd.read_csv(data_path+'processed/train.csv')
    test = pd.read_csv(data_path+'processed/test.csv')
    print('Train: ',train.shape)
    print('Test: ',test.shape)


    train_blast = pd.read_csv(data_path+'features/blast/processed/train.csv')
    test_blast = pd.read_csv(data_path+'features/blast/processed/test.csv')
    print('Blast Train: ',train_blast.shape)
    print('Test: ',test_blast.shape)

    scaler = RobustScaler()
    df = pd.concat([train_blast,test_blast],axis=0).iloc[:,:-1].astype(np.float32)
    scaler.fit(df)
    print('Scaler fitted')

    df = pd.DataFrame(scaler.transform(train_blast.iloc[:,:-1].astype(np.float32)))
    df['sequence_id'] = train_blast['sequence_id']
    train = pd.merge(train,df,how='left',on='sequence_id')

    df = pd.DataFrame(scaler.transform(test_blast.iloc[:,:-1].astype(np.float32)))
    df['sequence_id'] = test_blast['sequence_id']
    test = pd.merge(test,df,how='left',on='sequence_id')

    print('Train: ',train.shape)
    print('Test: ',test.shape)

    tokenizer = RobertaTokenizerFast.from_pretrained(data_path+"features/bert/")
    print(tokenizer.vocab_size())


if __name__ == '__main__':
    
    main()