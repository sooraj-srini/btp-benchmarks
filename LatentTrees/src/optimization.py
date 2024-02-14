import numpy as np
import torch

import torch.nn.functional as F

from tqdm import tqdm

def train_batch(x, y, LT_model, optimizer, criterion, nb_iter=1e4, monitor=None):

    n, d = x.shape
    
    # cast to pytorch Tensors
    t_y = torch.from_numpy(y[:, None]).float()
    t_x = torch.from_numpy(x).float()

    LT_model.train()

    pbar = tqdm(range(int(nb_iter)))
    for i in pbar:

        # print(LT_model.latent_tree.eta.detach().numpy())
        optimizer.zero_grad()

        y_pred = LT_model(t_x)

        loss = criterion(y_pred, t_y)

        pbar.set_description("train loss %s" % loss.detach().numpy())

        loss.backward()
        
        optimizer.step()

        if monitor:
            monitor.write(LT_model, i, report_tree=True, train={"Loss": loss.detach()})

def train_stochastic(dataloader, model, optimizer, criterion, epoch, monitor=None, prog_bar=True):

    model.train()

    last_iter = epoch * len(dataloader)

    train_obj = 0.

    if prog_bar:
        pbar = tqdm(dataloader)
    else:
        pbar = dataloader

    for i, batch in enumerate(pbar):

        # import pdb; pdb.set_trace()
        optimizer.zero_grad()

        t_x, t_y = batch

        if t_y.dim() > 2: # predictors support only flatten output atm
            t_y = t_y.view(len(t_y), -1)

        y_pred = model(t_x).squeeze()
        
        loss = criterion(y_pred, t_y) / len(t_x)
            
        train_obj += loss.detach().numpy()

        if prog_bar:
            pbar.set_description("avg train loss %f" % (train_obj / (i + 1)))
        loss.backward()

        optimizer.step()
        
        if monitor:
            monitor.write(model, i + last_iter, report_tree=True, train={"Loss": loss.detach()})
            
def evaluate(dataloader, model, criteria, epoch=None, monitor=None):

    model.eval()

    total_losses = {k: 0. for k in criteria.keys()}
    
    num_points = 0
    for batch in dataloader:

        t_x, t_y = batch
        
        if t_y.dim() > 2: # predictors support only flatten output atm
            t_y = t_y.view(len(t_y), -1)
        
        num_points += len(t_x)

        y_pred = model.predict(t_x).squeeze()

        for k in criteria.keys():
            loss = criteria[k](y_pred, t_y)
            total_losses[k] += loss.detach()

    if monitor:
        monitor.write(model, epoch, val={k: loss / num_points for k, loss in total_losses.items()})

    return {k: loss.numpy() / num_points for k, loss in total_losses.items()}

def train_ndf(dataloader, model, optimizer, epoch, jointly_training):

    # Update \Pi
    if not jointly_training:
        print("Epoch %d : Two Stage Learing - Update PI" % (epoch))
        # prepare feats
        cls_onehot = torch.eye(model.num_classes)
        feat_batches = []
        target_batches = []

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):

                # Get feats
                feats = model.feature_layer(data)
                feats = feats.view(feats.size()[0], -1)
                feat_batches.append(feats)
                target_batches.append(cls_onehot[target])

            # Update \Pi for each tree
            for tree in model.forest.trees:
                mu_batches = []
                for feats in feat_batches:
                    mu = tree(feats)  # [batch_size,n_leaf]
                    mu_batches.append(mu)
                for _ in range(20):
                    new_pi = torch.zeros((tree.n_leaf, tree.n_class))  # Tensor [n_leaf,n_class]
                    for mu, target in zip(mu_batches, target_batches):
                        pi = tree.get_pi()  # [n_leaf,n_class]
                        prob = tree.cal_prob(mu, pi)  # [batch_size,n_class]

                        pi = pi.data
                        prob = prob.data
                        mu = mu.data

                        _target = target.unsqueeze(1)  # [batch_size,1,n_class]
                        _pi = pi.unsqueeze(0)  # [1,n_leaf,n_class]
                        _mu = mu.unsqueeze(2)  # [batch_size,n_leaf,1]
                        _prob = torch.clamp(prob.unsqueeze(1), min=1e-6, max=1.)  # [batch_size,1,n_class]

                        _new_pi = torch.mul(torch.mul(_target, _pi), _mu) / _prob  # [batch_size,n_leaf,n_class]
                        new_pi += torch.sum(_new_pi, dim=0)

                    new_pi = F.softmax(new_pi, dim=1).data
                    tree.update_pi(new_pi)

    # Update \Theta
    model.train()

    for data, target in tqdm(dataloader):

        optimizer.zero_grad()

        output = model(data)
        
        loss = F.nll_loss(torch.log(output), target)

        loss.backward()

        optimizer.step()
