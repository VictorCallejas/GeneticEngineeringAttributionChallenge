import pandas as pd
import numpy as np

import torch

import os
from datetime import datetime
from shutil import copyfile
import sys

from data import get_test

DIR_PATH = '../artifacts/'
RUN_TYPE = None
FOLDER_NAME = None
RUN_FOLDER = None

class Logger():
    def __init__(self,path):
        self.terminal = sys.stdout
        self.log = open(path,mode='x')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        pass 

def config_run_logs():

    global RUN_TYPE

    RUN_TYPE = cfg.run['setup']

    now = datetime.now()

    global FOLDER_NAME
    FOLDER_NAME = now.strftime("%d_%m_%Y_%H_%M")

    global RUN_FOLDER

    RUN_FOLDER = DIR_PATH+RUN_TYPE+'/'+FOLDER_NAME

    try:
        os.makedirs(RUN_FOLDER)
    except:
        error = True
        print('Folder already exists')
        i = 2
        while error:
            RUN_FOLDER = RUN_FOLDER + str(i)
            try:
                os.makedirs(RUN_FOLDER)
                error = False
            except:
                i+=1

    print('Run folder created: ', RUN_FOLDER)
    redirect_output()

def redirect_output():
    try:
        sys.stdout = Logger(path=RUN_FOLDER+'/out.log')
    except:
        print('Log File exists')
    print('BASH OUTPUT REDIRECTED TO LOG FILE')


def save_model(model,epoch):

    if not os.path.isdir(RUN_FOLDER+'/model'):
        os.makedirs(RUN_FOLDER+'/model')

    path = RUN_FOLDER+'/model/'+str(epoch)+'.ckpt'
    torch.save(model,path)

    #print('Model saved on ',path)

def get_best_model(training_stats):
    ind = np.argmax(training_stats[['dev_auc']])
    path = RUN_FOLDER+'/model/'+str(ind)+'.ckpt'

    print('Best model is: ',path)

    best_model = torch.load(path)

    remove_other_models(ind)

    return best_model

def remove_other_models(best):
    base_path = RUN_FOLDER + '/model'
    for file in os.listdir(base_path):
        if file[0:-5] != str(best):
            path = base_path + '/' + file
            os.remove(path)
    print('Not best models deleted')

def make_submission(model, test_dataloader):

    print('MAKING SUBMISSION')

    model.eval()

    logits = []

    device = get_device()

    for batch in test_dataloader:

        with torch.no_grad():
          
            b_logits, _ = model(batch,device)

            logits.extend(b_logits)

    y_probas = torch.nn.Sigmoid()(torch.Tensor(logits))
    y_labels = y_probas.round()

    col_names = ['id','proba','label']

    subm = pd.DataFrame(columns=col_names)
    subm.id = get_test().id
    subm.label = y_labels.detach().numpy().astype(np.int64)
    subm.proba = y_probas.detach().numpy().astype(np.float64)

    print(subm.shape)
    print(subm.head())

    file_path = RUN_FOLDER+'/submission.csv'
    subm.to_csv(file_path,index = False, header=True,sep=',',encoding='utf-8-sig')

    print('SUBMISSION  DONE.')

def get_device():

    if torch.cuda.is_available() & (cfg.run['device'] == 'cuda'):    
    
        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    return device