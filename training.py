import torch
import math

from dataclasses import dataclass

from .utils import prediction, utils
from typing import Callable, Any, Dict, Tuple
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

try:
    from apex import amp
except:
    amp = None

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

StepCallback = Callable[[int, int, tqdm, torch.nn.Module,
                         torch.FloatTensor, Any,
                         torch.FloatTensor, torch.FloatTensor], None]

EpochCallback = Callable[[int, int, torch.nn.Module, tqdm], None]

MetricCall = Callable[[torch.FloatTensor, Any], Any]

LossCall = Callable[[torch.FloatTensor, Any], torch.FloatTensor]


def basic_step(model: torch.nn.Module, data: Tuple[torch.FloatTensor, Any], loss: LossCall, feed_target: bool = False, data_log = {}):
    input, target = data

    input = input.to(device)
    target = target.to(device)

    pred = model(input, target) if feed_target else model(input)
    cost = loss(pred, target)

    return cost, input, target, pred


def training_step(cost: torch.FloatTensor, optimizer: torch.optim.Optimizer, fp16: bool = False):
    optimizer.zero_grad()

    if fp16:
        with amp.scale_loss(cost, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        cost.backward()
    
    optimizer.step()


@dataclass
class OptimParams:
    epochs: int
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    loss: LossCall
    dl_train: DataLoader
    dl_valid: DataLoader = None
    metric: MetricCall = None
    ittr_limit: int = None
    lr_scheduler: _LRScheduler = None
    feed_target: bool = False
    batch_dim: int = 0
    bare_train_feedback: bool = False
    print_result: bool = False
    epoch_save_path: str = None
    direct_use_dl: bool = False
    fp_16: bool = False
    data_log_dict = {}

def optimize_with_params(params: OptimParams, basic_stepper = basic_step, training_stepper = training_step, step_callback = None, epoch_callback = None):
    if params.bare_train_feedback:
        return optimize_bare(
            params.epochs, params.model, params.optimizer, params.loss,
            params.dl_train, params.feed_target, basic_stepper, 
            training_stepper, params.lr_scheduler, params.ittr_limit, 
            params.data_log_dict, params.direct_use_dl, params.fp_16)

    else:
        return optimize(
            params.epochs, params.model, params.optimizer, params.loss,
            params.dl_train, params.dl_valid, params.metric, params.feed_target,
            epoch_callback, step_callback, params.print_result,
            params.batch_dim, training_stepper, params.ittr_limit, params.lr_scheduler,
            params.epoch_save_path, basic_stepper, params.fp_16)

def optimize_bare(epochs: int,
                  model: torch.nn.Module,
                  optimizer: torch.optim.Optimizer,
                  loss: LossCall,
                  dl: DataLoader,
                  feed_target: bool = False,
                  basic_stepper=basic_step,
                  training_stepper=training_step,
                  lr_scheduler = None,
                  ittr_limit = None,
                  data_log_dict = {},
                  direct_use_dl = False,
                  fp16 = False): # implement last two params in optimize as well; make sure dl is iter in basic info loop not enum

    losses = []
    bar = utils.tqdm(range(epochs)) if epochs > 1 else [0]
    model.train()

    if fp16:
        model, optimizer = amp.initialize(model, optimizer)

    for _ in bar:
        dl = dl if direct_use_dl and epochs == 1 else iter(dl)

        for data in dl:
            if lr_scheduler is not None:
                lr_scheduler.step()
            
            cost, _, _, _ = basic_stepper(model, data, loss, feed_target, data_log_dict)
            training_stepper(cost, optimizer, fp16)

            losses.append(cost.item())

            if ittr_limit is not None and len(losses) >= ittr_limit:
                return False, losses, dl

    return True, losses, None

def optimize(epochs: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
             loss: LossCall,
             train_dl: DataLoader,
             valid_dl: DataLoader = None,
             metric: MetricCall = None,
             feed_target: bool = False,
             epoch_callback: EpochCallback = None,
             step_callback: StepCallback = None,
             print_results: bool = True,
             batch_dim: int = 0,
             training_stepper=training_step,
             ittr_limit: int = None,
             lr_scheduler = None,
             epoch_save_path: str = None,
             basic_stepper = basic_step,
             fp16 = False):

    epoch_ittr = utils.tqdm(range(epochs))
    data = {"train": [], "test": []}
    model.train()

    if fp16:
        model, optimizer = amp.initialize(model, optimizer)

    for epoch in epoch_ittr:
        epoch_ittr.write("\nEpoch: " + str(epoch))
        
        if epoch_callback is not None:
            epoch_callback(epoch, epochs, model, epoch_ittr)
        
        train_data = train(model, optimizer, loss, train_dl, feed_target,
                           callback=step_callback, stepper=training_stepper, 
                           basic_stepper=basic_stepper, ittr_limit=ittr_limit, 
                           lr_scheduler=lr_scheduler, fp16=fp16)

        if epoch_save_path is not None:
            torch.save(model.state_dict(), f"{epoch_save_path}_{str(epoch)}_model.pt")
            torch.save(optimizer.state_dict(), f"{epoch_save_path}_{str(epoch)}_optim.pt")

            if lr_scheduler is not None:
                torch.save(lr_scheduler.state_dict(), f"{epoch_save_path}_{str(epoch)}_lrs.pt")

        eval_data = None

        if valid_dl is not None:
            eval_data = evaluate(model, loss, valid_dl, batch_dim=batch_dim, 
                                 callback=step_callback, basic_stepper=basic_stepper,
                                 store_pt=metric is not None)
       
            if metric is not None:
                eval_data["metric"] = metric(eval_data['pred'], eval_data['target'])
                
                del eval_data['pred']
                del eval_data['target']

        if print_results:
            print_info(epoch_ittr.write, train_data, eval_data)

        data["train"] += [train_data]
        data["test"] += [eval_data]

        if epoch_save_path is not None:
            torch.save({"train": train_data, "test": eval_data}, f"{epoch_save_path}_{str(epoch)}_data.pt")
            
    return data


def print_info(write_fn: Callable[[str], None],
               train_data: Dict[str, Any] = None,
               eval_data: Dict[str, Any] = None):

    if train_data is not None:
        cost = train_data['cost']
        mean = sum(cost) / len(cost)
        
        write_fn("Train Cost: " + str(mean))
    
    if eval_data is not None:
        write_fn("Test Cost: " + str(eval_data["cost"]))

    if eval_data is not None and "metric" in eval_data:
        write_fn("Metric: " + str(eval_data["metric"]))


def train(model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss: LossCall,
          train_dl: DataLoader, feed_target: bool = False, callback: StepCallback = None, 
          stepper=training_step, basic_stepper=basic_step, ittr_limit: int = None, lr_scheduler = None, fp16 = False):

    if lr_scheduler is not None:
        data = {'cost': [], 'lrs': []}
    else:
        data = {'cost': []}

    def step(x, y, pred, cost, i, steps, bar, model):  
        if callback is not None:
            callback(i, steps, bar, model, x, y, pred, cost)
        
        data['cost'].append(cost.cpu().detach().item())
        
        if lr_scheduler is not None:
            data['lrs'].append(optimizer.state_dict()["param_groups"][0]["lr"])

        stepper(cost, optimizer, fp16=fp16)

        if lr_scheduler is not None:
            lr_scheduler.step()

    basic_info_loop(model, loss, train_dl, step, feed_target=feed_target, stepper=basic_stepper, ittr_limit=ittr_limit)
        
    return data


@prediction.eval
def evaluate(model: torch.nn.Module, loss: LossCall, valid_dl: DataLoader, batch_dim: int = 0,
        callback: StepCallback = None, basic_stepper=basic_step, store_pt:bool = True):

    data = {'cost': 0, 'pred': [], 'target': []}
    
    def step(x, y, pred, cost, i, steps, bar, model):           
        if callback is not None:
            callback(i, steps, bar, model, x, y, pred, cost)
        
        data['cost'] += cost.cpu().detach().item()

        if store_pt:
            data['pred'].append(pred.cpu().detach())
            data['target'].append(y.cpu().detach())

    basic_info_loop(model, loss, valid_dl, step, stepper=basic_stepper)
    
    data["cost"] /= len(valid_dl)
    
    if store_pt:
        data['pred'] = torch.cat(data['pred'], dim=batch_dim)
        data['target'] = torch.cat(data['target'], dim=batch_dim)

    return data


def basic_info_loop(model: torch.nn.Module,
                    loss: LossCall,
                    dl: DataLoader,
                    step: Callable[[torch.FloatTensor, Any,
                                    torch.FloatTensor, torch.FloatTensor,
                                    int, int, tqdm, torch.nn.Module], None],
                    feed_target: bool = False,
                    stepper=basic_step,
                    ittr_limit=None):

    steps = len(dl)

    if ittr_limit is not None:
        steps = min(steps, ittr_limit)
     
    if steps < 10:
        exp_cost_factor = 1
    else:
        exp_cost_factor = 1 / (steps // 10)
        
    moving_av_cost = 0
    exp_weighted_cost = 0 
    
    bar = utils.tqdm(enumerate(dl), total=steps)

    for i, data in bar:
        cost, input, target, pred = stepper(model, data, loss, feed_target)

        exp_weighted_cost = (1 - exp_cost_factor) * exp_weighted_cost + exp_cost_factor * cost.item()
        moving_av_cost += cost.item()

        bias_correction = 1 - (1 - exp_cost_factor) ** (i + 1)

        bar.set_postfix({
            'cost': round(cost.item(), 4),
            'avg_cost': round(moving_av_cost / (i + 1), 4),
            'exp_cost': round(exp_weighted_cost / bias_correction, 4)
        })

        step(input, target, pred, cost, i, steps, bar, model)
        
        if ittr_limit is not None and i >= ittr_limit:
            break

            
def lr_range(dl, model, loss, optim_fn, mn: float = 1e-6, mx: float = 1, linear: bool = False, ittrs: int = None, **kwargs):
    steps = len(dl)
    ittr_limit = None
    epochs = 1
    
    if linear:
        mn = 1
    
    if ittrs is not None:
        if ittrs < steps:
            steps = ittrs
            ittr_limit = ittrs
        elif ittrs > steps:
            epochs = ittrs // steps
        
    optim = optim_fn(model.parameters(), mn)
    
    if linear:
        schedual = torch.optim.lr_scheduler.LambdaLR(optim, lambda t: (t/steps) * mx)
    else:
        schedual = torch.optim.lr_scheduler.ExponentialLR(optim, math.pow(mx/mn, 1/steps))
    
    return optimize(
        epochs=epochs,
        model=model, 
        optimizer=optim,
        loss=loss, 
        train_dl=dl,
        ittr_limit=ittr_limit,
        lr_scheduler=schedual,
        **kwargs
    )

