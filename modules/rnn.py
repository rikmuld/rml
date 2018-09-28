import torch

from torch import nn
from ..utils.utils import device
from ..utils import text
from typing import Tuple, List

import random


class InitHidden(nn.Module):
    def __init__(self, rnn: nn.RNNBase, num_layers: int, hidden_size: int, bidirectional: bool = False):
        super(InitHidden, self).__init__()
        
        self.rnn = rnn
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        if bidirectional:
            self.num_layers *= 2
        
    def forward(self, x: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        bs = x.size()[1]
        hidden = self.init_hidden(bs)
        
        return self.rnn(x, hidden)
    
    def init_hidden(self, bs: int) -> torch.FloatTensor:
        return torch.zeros(self.num_layers, bs, self.hidden_size).to(device)


class InitSequence(nn.Module):
    def __init__(self, rnn: nn.RNNBase, output_length: int, pass_context: bool = False,
                 attention_input_size: int = None, attention_size: int = None):

        super(InitSequence, self).__init__()
        
        if pass_context:
            self.attention = Attention(attention_input_size, attention_size)
            
        self.rnn = rnn
        self.output_length = output_length
        self.teacher_forcing_prob = 1
    
    def forward(self, hidden: torch.FloatTensor,
                enc_output: torch.FloatTensor = None,
                targets: torch.LongTensor = None) -> Tuple[torch.FloatTensor, torch.FloatTensor]:

        bs = hidden.size()[1]
        x = InitSequence.init_sequence(bs)
        
        ys = None
            
        for i in range(self.output_length):            
            y, hidden = self.forward_step(x, hidden, enc_output)
            
            ys = self.store_result(i, bs, y, ys)
            x = self.get_next_input(i, y, targets)                      

        return ys, hidden
    
    def forward_step(self, x: torch.FloatTensor,
                     hidden: torch.FloatTensor,
                     enc_output: torch.FloatTensor = None) -> Tuple[torch.FloatTensor, torch.FloatTensor]:

        if self.attention is not None:
            return self.rnn(x, hidden, self.attention(enc_output, hidden))
        else:
            return self.rnn(x, hidden)
        
    def store_result(self, i: int, bs: int, y: torch.FloatTensor, ys: torch.FloatTensor) -> torch.FloatTensor:
        ys = ys if ys is not None else y.new_empty((self.output_length, bs, y.size()[-1]))
        ys[i] = y
        
        return ys
    
    def get_next_input(self, i: int, y: torch.FloatTensor, targets: torch.LongTensor=None) -> torch.LongTensor:
        no_teacher_forcing = targets is None or i >= len(targets) or self.teacher_forcing_prob < random.random()
        
        if no_teacher_forcing:
            return y.data.max(2)[1]
        else:
            return targets[i].unsqueeze(0)

    @staticmethod
    def init_sequence(bs: int) -> torch.FloatTensor:
        return torch.zeros((1, bs)).long().to(device)


class Attention(nn.Module):
    store = None
    
    def __init__(self, input_size: int, attention_size: int):
        super(Attention, self).__init__()
        
        self.lin1 = nn.Linear(input_size, attention_size)
        self.tanh = nn.Tanh()
        self.lin2 = nn.Linear(attention_size, 1)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x: torch.FloatTensor, hidden: torch.FloatTensor) -> torch.FloatTensor:
        hidden = text.merge_hidden_from_layers(hidden)
                
        seq_len, bs, out_size = x.size()
        _, _, hidden_size = hidden.size()
        
        hidden = hidden.expand((seq_len, bs, hidden_size))
        
        e = torch.cat((x, hidden), dim=-1)
        e = self.lin1(e)
        e = self.tanh(e)
        e = self.lin2(e)
        
        a = self.softmax(e)
        
        if Attention.store is not None:
            Attention.store.append(a)
        
        c = x * a
        c = c.sum(0, keepdim=True)
        
        return c
    
    @staticmethod
    def set_capture(lst: List[torch.FloatTensor] = None) -> None:
        Attention.store = lst
        
    @staticmethod
    def stop_capture() -> None:
        Attention.store = None
        
    @staticmethod
    def get_capture() -> List[torch.FloatTensor]:
        return Attention.store
