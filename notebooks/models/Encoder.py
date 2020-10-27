import torch
import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self,b_e,t_e,n_e):

        super().__init__()

        self.idx_end_blast = 40 + 6570
        self.idx_end_tok = self.idx_end_blast + 5000

        self.blast = torch.load('artifacts/BlastEncoder/'+str(b_e)+'.ckpt')
        self.tok = torch.load('artifacts/TokEncoder/'+str(t_e)+'.ckpt')
        self.ngram = torch.load('artifacts/NGramEncoder/'+str(n_e)+'.ckpt')
        
    def forward(self,batch, device):

        inputs = batch[0]
        x0 = inputs[:,:40].to(device)
        x1 = inputs[:,40:self.idx_end_blast].to(device)
        x2 = inputs[:,self.idx_end_blast:self.idx_end_tok].to(device)
        x3 = inputs[:,-3905:].to(device)

        x1 = self.blast(x1)
        x2 = self.tok(x2)
        x3 = self.ngram(x3)

        x = torch.cat((x0,x1,x2,x3),dim=1)

        return x.detach().cpu().float().numpy()
    