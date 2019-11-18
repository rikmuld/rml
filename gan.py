from .training import *

import numpy as np


def build_gan_stepper(D_model, D_transform, adv_loss, adv_loss_weight=1, replay_buffer_size=25):
    def gan_basic_stepper(model: torch.nn.Module, data: Tuple[torch.FloatTensor, Any], loss: LossCall, feed_target: bool = False, data_log_dict = {}):
        if "disc_inputs" not in data_log_dict:
            disc_inputs = []
        else:
            disc_inputs = data_log_dict["disc_inputs"]

        if len(disc_inputs) >= replay_buffer_size:
            disc_inputs = disc_inputs[1:]

        cost, input, target, pred = basic_step(model, data, loss, feed_target, data_log_dict)
        
        D_input_act, D_input_pred = D_transform(input, target, pred)
        D_target_act, D_target_pred = torch.ones((target.size(0))), torch.zeros((target.size(0))) 

        D_model.eval()
        adv_cost = adv_loss_weight * adv_loss(D_model(D_input_pred)) 
        D_model.train()

        perm = torch.LongTensor(np.random.permutation(list(range(len(D_input_act) * 2)))).cuda()

        D_input = torch.cat([D_input_act, D_input_pred])[perm]
        D_targets = torch.cat([D_target_act, D_target_pred]).unsqueeze(1).to(utils.device)[perm]
        
        data_log_dict["disc_inputs"] = disc_inputs + [(D_input.detach(), D_targets)]
        data_log_dict["cost"] = cost.item()
        data_log_dict["cost_adv"] = adv_cost.item()

        return cost + adv_cost, input, target, pred

    return gan_basic_stepper


# make work with optimize (just only support optimize bare)
def optimize_gan(
    itterations: int, G_params: OptimParams, D_params: OptimParams, adv_loss, D_ds_transform, 
    adv_loss_weight=1, replay_buffer_size=25, 
    G_loss_threshold=None, D_loss_threshold=None):

    gen_stepper = build_gan_stepper(D_params.model, D_ds_transform, adv_loss, adv_loss_weight, replay_buffer_size)
    G_dl = G_params.dl_train

    data_log_dict = {}
    G_params.data_log_dict = data_log_dict
    G_params.direct_use_dl = True
    G_params.dl_train = iter(G_dl)

    lG, lD, lGa, lGf = [], [], [], []    
    bar = utils.tqdm(range(itterations))

    for i in bar:
        train_G = len(lG) == 0 or G_loss_threshold is None or np.mean(lG[-25:]) > G_loss_threshold
        train_G = training_step if train_G else lambda *x:None

        epoch_finished, losses, _ = optimize_with_params(G_params, basic_stepper=gen_stepper, training_stepper=train_G)
        
        if epoch_finished:
            G_params.dl_train = iter(G_dl)

        G_loss = np.mean(losses)
        G_loss_feature = data_log_dict["cost"]
        G_loss_adv = data_log_dict["cost_adv"]

        train_D = len(lD) == 0 or D_loss_threshold is None or np.mean(lD[-25:]) > D_loss_threshold
        train_D = training_step if train_D else lambda *x:None
 
        D_params.dl_train = np.random.permutation(data_log_dict["disc_inputs"])
        _, losses, _ = optimize_with_params(D_params, training_stepper=train_D)
        D_loss = np.mean(losses)

        lG.append(G_loss); lGa.append(G_loss_adv); lGf.append(G_loss_feature); lD.append(D_loss)
        
        bar.set_postfix({
            "G_feature": round(np.mean(lGf[-25:]), 4),
            "G_adv": round(np.mean(lGa[-25:]), 4),
            "G": round(np.mean(lG[-25:]), 4),
            "D": round(np.mean(lD[-25:]), 4)
        })

        # remove below
        if i % 1000 == 0:
            torch.save((G_params.model.state_dict(), D_params.model.state_dict()), f"model_store/GAN_{i}.pt")
            torch.save((lG[-1000:], lD[-1000:], lGa[-1000:], lGf[-1000:]), f"model_store/data_{i}.pt")

    torch.save((G_params.model.state_dict(), D_params.model.state_dict()), "model_store/GAN_final.pt")
    torch.save((lG, lD, lGa, lGf), "model_store/data_full.pt")

    return lG, lD, lGa, lGf