import torch.nn as nn
from topk.svm import SmoothTopkSVM, MaxTopkSVM
import torch

class BlastEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.out_shape = 1314

        self.stride = 5
        self.c1_out = 2
        self.num_feats = 5

        self.hidden_size = self.c1_out*self.out_shape
        self.latent_size = 200

        self.c1 = nn.Conv1d(1, self.c1_out, self.num_feats, self.stride, padding = 0)
        self.bc1 = nn.BatchNorm1d(self.hidden_size)

        self.l1 = nn.Linear(self.hidden_size,1000)
        self.b1 = nn.BatchNorm1d(1000)
        self.a1 = nn.LeakyReLU()

        self.l2 = nn.Linear(1000,500)
        self.b2 = nn.BatchNorm1d(500)
        self.a2 = nn.LeakyReLU()

        self.l3 = nn.Linear(500,self.latent_size)

    def forward(self,x):

        x = self.c1(x.unsqueeze(1)).flatten(1)
        x = self.bc1(x)

        x = self.l1(x)
        x = self.b1(x)
        x = self.a1(x)

        x = self.l2(x)
        x = self.b2(x)
        x = self.a2(x) 

        x = self.l3(x)

        return x

class BlastDecoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.out_shape = 1314

        self.stride = 2
        self.c1_out = 2
        self.num_feats = 5

        self.hidden_size = self.c1_out*self.out_shape
        self.latent_size = 200

        self.bl = nn.BatchNorm1d(self.latent_size)
        self.l1 = nn.Linear(self.latent_size,self.out_shape)


    def forward(self,x):

        x = self.bl(x)
        x = self.l1(x)

        return x

class Blast(nn.Module):

    def __init__(self):
        super().__init__()

        self.folder = 'artifacts/BlastEncoder/'

        self.input_shape = 6570

        self.idx_start_blast = 40
        self.idx_end_blast = self.idx_start_blast + self.input_shape

        self.encoder = BlastEncoder()
        self.decoder = BlastDecoder()

        self.criterion = MaxTopkSVM(1314,k=10,alpha=.5).cuda()
        
    def forward(self,batch, device):

        inputs = batch[0]
        b_labels = batch[1].to(device)
        inputs = inputs[:,self.idx_start_blast:self.idx_end_blast].to(device)

        x = self.encoder(inputs)
        out = self.decoder(x)

        loss = self.criterion(out,torch.argmax(b_labels,dim=1))

        return x, loss


