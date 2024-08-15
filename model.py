import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

class NeuralCell(nn.Module):
    def __init__(self,in_features=1, out_features=1, bias=True) -> None:
        super(NeuralCell,self).__init__()
        self.f = nn.Linear(in_features, out_features, bias=bias)

    def _init(self, w=0.01):
        nn.init.constant_(self.f.weight, w)
        nn.init.zeros_(self.f.bias)

    def setweight(self, weight, bias):
        with torch.no_grad():
            self.f.weight.copy_(weight)
            self.f.bias.copy_(bias)

    def getweight(self):
        return self.f.weight, self.f.bias

    def culculatepassthrough(self):
        w,b = self.getweight()
        return torch.abs(w+b).mean()

    def forward(self,x):
        return self.f(x)

class NeuralCellEdge(nn.Module):
    def __init__(self,in_features=1, out_features=1, bias=True, init_w=None) -> None:
        super(NeuralCellEdge,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.fs = nn.ModuleList()
        self.device = 'cpu'

        if init_w is not None:
            self._init(init_w)
        
    def _init(self,w):
        cell = NeuralCell(in_features=self.in_features, out_features=self.out_features, bias=self.bias)
        self.fs.append(cell)
        for f in self.fs:
            f._init(w)

    def to(self,device):
        self.device = device
        for f in self.fs:
            f.to(device)
        return self
    
    def __str__(self):
        return f'{len(self.fs)}'
    
    def _show_weight(self):
        for f in self.fs:
            weight, bias = f.getweight()
            print(weight, bias)

    def culculatepassthrough(self):
        passts = []
        for f in self.fs:
            passts.append(f.culculatepassthrough())
        return torch.stack(passts)
    
    def create(self, idx): # create new neural cell
        try:
            w,b = self.fs[idx].getweight()
            self.fs[idx].setweight(w/2,b/2)
            newf = NeuralCell(
                in_features=self.in_features,
                out_features=self.out_features,
                bias=self.bias,
                ).to(self.device)
            newf.setweight(w/2,b/2)
            self.fs.append(newf)
        except Exception as e:
            print(f'Create {idx} cell failed',e)

    def delete(self, idx): # delete neural cell
        try:
            if len(self.fs) > 1:
                del self.fs[idx]
            # self.fs[idx].pop(idx)
        except Exception as e:
            print(f'delete {idx} cell failed',e)

    def update(self,create_list, delete_list): # update the neural cell list, create or delete some neural cells base on some rule
        for c in sorted(create_list):
            self.create(c)
        for d in sorted(delete_list, reverse=True):
            self.delete(d)

    def forward(self,x):
        y = 0
        for f in self.fs:
            y += f(x)
        return y

class Brain(nn.Module):
    def __init__(self,in_features=784, out_features=10) -> None: # default 784 ,10 is Mnist dataset input output
        super(Brain,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.init_w = 1/out_features
        self.edges = nn.ModuleList()
        self.device = 'cpu'
        
    def _init(self):
        del self.edges
        self.edges = nn.ModuleList()
        for i in range(self.in_features):
            self.edges.append(nn.ModuleList([NeuralCellEdge(1, 1, True, init_w=1/self.out_features) for _ in range(self.out_features)]))
    
    def to(self,device):
        self.device = device
        for es in self.edges:
            for e in es:
                e.to(device)
        return self
    
    def extinction(self, min_th=None, max_th=None):
        if min_th is None:
            min_th = self.init_w/2
        if max_th is None:
            max_th = self.init_w*2
        for es in self.edges:
            for e in es:
                tps = e.culculatepassthrough()
                create_list = torch.nonzero(tps>=max_th, as_tuple=True)[0]
                delete_list = torch.nonzero(tps<min_th, as_tuple=True)[0]
                e.update(create_list,delete_list)
        self.init_w = self.init_w/2

    def load(self,path):
        pass
    
    def _get_arch(self):
        table = np.zeros((len(self.edges), self.out_features), dtype=int)
        for i in range(len(self.edges)):
            for o,e in enumerate(self.edges[i]):
                table[i,o] = str(e)
        return table

    def __str__(self):
        # rstr = 'i->e->o\n'
        # for i in range(len(self.edges)):
        #     for o,e in enumerate(self.edges[i]):
        #         rstr+=f'{i}->{e}->{o}|'
        #     rstr+='\n'
        # return rstr
        table = pd.DataFrame(self._get_arch())
        return table.to_string()
    
    def _show_weight(self):
        for es in self.edges:
            for e in es:
                e._show_weight()

    def forward(self,x, act=F.sigmoid):
        sp = tuple(x.shape[:-1])+(self.out_features,)
        ys = torch.zeros(*sp,device=x.device)
        for es in self.edges:
            for i in range(len(es)):
                ys[...,i:i+1] += es[i](x[...,i:i+1])
        return act(ys)
    

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = Brain(in_features=784, out_features=10)
    m._init()
    m.to(device)
    x = torch.rand(10, 784).to(device)
    y = m(x)
    print(m)
    print(y.shape)
    print(y)        
    # update test
    m.extinction(max_th=1/785)
    y = m(x)
    print(m)
    print(y.shape)
    print(y)