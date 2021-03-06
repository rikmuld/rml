import torch

from .utils import device, tqdm
from typing import Any, Callable

from functools import reduce
from operator import add

def eval(f):
    def wrapper(model: torch.nn.Module, *args, **kwargs):
        model.eval()
        
        with torch.no_grad():
            result = f(model, *args, **kwargs)

        model.train()
        
        return result
    
    return wrapper

def to_device(inp):
    return inp.to(device)

@eval
def predict_one(model: torch.nn.Module, inp, transform: Callable[[Any], torch.FloatTensor]=None, 
                batch_dim: int = 0) -> torch.FloatTensor:
    if transform:
        inp = transform(inp)

    inp = inp.unsqueeze(batch_dim)
    inp = inp.to(device)

    return model(*inp)

@eval
def predict_many(model: torch.nn.Module, inp: Any, bs: int = -1, batch_dim: int = 0,
                 to_device_fn=to_device) -> torch.FloatTensor:
    inp = inp.to(device)
    res = []
    
    for lower in tqdm(range(0, len(inp), bs)):
        batch = inp[lower:min(len(inp), lower + bs)]
        res.append(model(batch))

    return torch.cat(res, dim=batch_dim)

@eval
def predict_dl(model: torch.nn.Module, dl: torch.utils.data.DataLoader, to_device_fn=to_device, loss_fn=None) -> (torch.FloatTensor, Any):
    pred, targ = [], []
    losses = []
    target = None

    for batch in tqdm(dl):
        data = batch
        
        if type(data) is tuple or type(data) is list:
            inp, target = data
        else:
            inp = data

        out = model(*to_device_fn(inp))

        if loss_fn is None:
            pred.append(out)
            targ.append(target)
        else:
            losses.append(loss_fn(out, *to_device_fn(target)))

    if loss_fn is None:
        if targ[0] is None:
            return torch.cat(pred)
        else:
            return torch.cat(pred), torch.cat(targ)
    else:
        return losses