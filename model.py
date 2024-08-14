import torch
import torch.nn as nn

class NeuralCell(nn.Module):
    def __init__(self,in_features=1, out_features=1, bias=True) -> None:
        super(NeuralCell,self).__init__()
        self.f = nn.Linear(in_features, out_features, bias=bias)

    def _init(self):
        self.f.weight.data.fill_(1)
        self.f.bias.data.zero_()

    def setweight(self, weight, bias):
        with torch.no_grad():
            self.f.weight.copy_(weight)
            self.f.bias.copy_(bias)

    def getweight(self):
        return self.f.weight, self.f.bias

    def forward(self,x):
        return self.f(x)

class NeuralCellEdge(nn.Module):
    def __init__(self,in_features=1, out_features=1, bias=True, is_init=True) -> None:
        super(NeuralCellEdge,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.fs = []
        self.device = 'cpu'

        if is_init:
            self._init()
        
    def _init(self):
        self.fs.append(NeuralCell(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=self.bias,
            ))
        for f in self.fs:
            f._init()

    def to(self,device):
        self.device = device
        for f in self.fs:
            f.to(device)
    
    def __str__(self):
        return f'{len(self.fs)}'
    
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
            del self.fs[idx]
            # self.fs[idx].pop(idx)
        except Exception as e:
            print(f'delete {idx} cell failed',e)

    def update(self,create_list, delete_list): # update the neural cell list, create or delete some neural cells base on some rule
        for c in create_list:
            self.create(c)
        for d in delete_list:
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
        self.edges = []
        self.device = 'cpu'

    def _init(self):
        for i in range(self.in_features):
            self.edges.append([NeuralCellEdge(1,1,True) for j in range(self.out_features)])
    
    def to(self,device):
        self.device = device
        for es in self.edges:
            for e in es:
                e.to(device)
    
    def extinction(self):
        pass

    def load(self,path):
        pass
    
    def __str__(self):
        rstr = 'i->e->o\n'
        for i in range(len(self.edges)):
            for o,e in enumerate(self.edges[i]):
                rstr+=f'{i}->{e}->{o}|'
            rstr+='\n'
        return rstr

    def forward(self,x):
        ys = [0.0]*self.out_features
        
        for es in self.edges:
            for i in range(len(es)):
                ys[i] += es[i](x[...,i:i+1])
        ys = torch.cat(ys, dim=-1)
        return ys
    

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = Brain(in_features=784, out_features=10)
    m._init()
    m.to(device)
    x = torch.rand(10, 784).to(device)

    # update test
    m.edges[0][0].update([0,1,2,3], [0,1])
    print(m)
    y = m(x)
    print(y.shape)