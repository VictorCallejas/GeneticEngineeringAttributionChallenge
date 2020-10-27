import torch.nn as nn
import torch

class TokEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.input_size = 5000
        self.latent_size = 500

        self.l1 = nn.Linear(self.input_size,3000)
        self.b1 = nn.BatchNorm1d(3000)

        self.l2 = nn.Linear(3000,1500)
        self.b2 = nn.BatchNorm1d(1500)

        
        self.l3 = nn.Linear(1500,900)
        self.b3 = nn.BatchNorm1d(900)

        self.a = nn.LeakyReLU()

        self.l4 = nn.Linear(900,self.latent_size)

    def forward(self,x):

        x = self.l1(x)
        x = self.b1(x)
        x = self.a(x)

        x = self.l2(x)
        x = self.b2(x)
        x = self.a(x) 

        x = self.l3(x)
        x = self.b3(x)
        x = self.a(x) 

        x = self.l4(x)

        return x

class TokDecoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.out_size = 5000
        self.latent_size = 500

        self.l1 = nn.Linear(self.latent_size,900)
        self.b1 = nn.BatchNorm1d(900)

        self.l2 = nn.Linear(900,1500)
        self.b2 = nn.BatchNorm1d(1500)

        
        self.l3 = nn.Linear(1500,3000)
        self.b3 = nn.BatchNorm1d(3000)

        self.a = nn.LeakyReLU()

        self.l4 = nn.Linear(3000,self.out_size)

    def forward(self,x):

        x = self.l1(x)
        x = self.b1(x)
        x = self.a(x)

        x = self.l2(x)
        x = self.b2(x)
        x = self.a(x) 

        x = self.l3(x)
        x = self.b3(x)
        x = self.a(x) 

        x = self.l4(x)

        return x


class Tok(nn.Module):

    def __init__(self):

        super().__init__()

        self.folder = 'artifacts/TokEncoder/'

        self.input_shape = 5000

        self.idx_start_tokens = 6570 + 40
        self.idx_end_tokens = self.idx_start_tokens + self.input_shape

        self.encoder = TokEncoder()
        self.decoder = TokDecoder()

        self.criterion = nn.MSELoss()
        
    def forward(self,batch, device):

        inputs = batch[0]
        inputs = inputs[:,self.idx_start_tokens:self.idx_start_tokens+5000].to(device)

        x = self.encoder(inputs)
        out = self.decoder(x)

        loss = self.criterion(out,inputs)

        return x, loss
    