# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import math
import numpy as np
import os
import time
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable

from model import Encoder, Decoder, D_Module
from common import *


def get_dynamics_losses(s, s_hat, s_prime, s_prime_hat, z_prime, z_prime_hat,
                        a_hat, curr_actions, discrete=False):
    # reconstruction loss
    recon_loss = F.mse_loss(s_hat, s)

    # next state prediction loss
    model_loss = F.mse_loss(s_prime_hat, s_prime)

    # net decoder loss
    dec_loss = (F.mse_loss(s_hat, s) + F.mse_loss(s_prime_hat, s_prime))

    # action reconstruction loss
    if discrete:
        a_hat = F.log_softmax(a_hat)
    inv_loss = F.mse_loss(a_hat, curr_actions)

    # representation space constraint
    forward_loss = F.mse_loss(z_prime_hat, z_prime.detach())
    return recon_loss, model_loss, dec_loss, inv_loss, forward_loss


def get_maze_dynamics_losses(s, s_hat_logits,
                        s_prime, s_prime_hat_logits,
                        z_prime, z_prime_hat,
                        a_hat_logits, curr_actions, discrete=True,
                        dec_mask=None):
    """
    dec_mask: if to reweigh the weights on the agent and goal locations,
    """
    # reconstruction loss
    if dec_mask is not None:
        recon_loss = F.cross_entropy(s_hat_logits.view(-1, 2), s.view(-1).long(), reduce=False)
        recon_loss = (recon_loss * dec_mask).mean()
    else:
        recon_loss = F.cross_entropy(s_hat_logits.view(-1, 2), s.view(-1).long())

    # next state prediction loss
    if dec_mask is not None:
        model_loss = F.cross_entropy(s_prime_hat_logits.view(-1, 2), s_prime.view(-1).long(), reduce=False)
        model_loss = (model_loss * dec_mask).mean()
    else:
        model_loss = F.cross_entropy(s_prime_hat_logits.view(-1, 2), s_prime.view(-1).long())

    # net decoder loss
    dec_loss = recon_loss + model_loss

    # action reconstruction loss
    inv_loss = F.cross_entropy(a_hat_logits, curr_actions.view(-1).long())

    # representation space constraint
    forward_loss = F.mse_loss(z_prime_hat, z_prime.detach())

    return recon_loss, model_loss, dec_loss, inv_loss, forward_loss

class DynamicsDataset(data.Dataset):
    def __init__(self, root, size, batch, rollout):
        self.size = size
        self.root = root
        self.actions = []
        self.states = []
        start = 0

        while len(self.actions) < size:
            end = start + batch
            states, actions = torch.load(
                os.path.join(self.root, 'states_actions_%s_%s.pt' % (start, end)))
            self.states += states
            self.actions += actions
            start = end
            rollout = len(actions[0])
        self.actions = torch.Tensor(self.actions[:size]).view(
            self.size, rollout, -1)
        self.states = torch.Tensor(self.states[:size]).view(
            self.size, rollout + 1, -1)

    def __getitem__(self, index):
        assert index < self.size
        return self.states[index], self.actions[index]

    def __len__(self):
        return len(self.actions)


class MazeDynamicsDataset(data.Dataset):
    def __init__(self, root, size, batch, rollout):
        """
        batch: is the size of the blocks of the data
        size: total size of the dataset, num of trajectories
        rollout: length of the trajectory
        """
        self.size = size
        self.root = root
        self.actions = []
        self.states = []
        start = 0

        while len(self.actions) < size:
            end = start + batch
            states, actions = torch.load(
                os.path.join(self.root, 'states_actions_%s_%s.pt' % (start, end)))
            self.states += states
            self.actions += actions
            start = end

        # convert the state and actions to the float
        self.states = np.asarray(self.states, dtype=np.float32)
        self.actions = np.asarray(self.actions, dtype=np.float32)

        # convert to tensors
        self.actions = torch.Tensor(self.actions).view(
            self.size, rollout, -1)
        self.states = torch.Tensor(self.states).view(
            self.size, rollout + 1, -1)

    def __getitem__(self, index):
        assert index < self.size
        return self.states[index], self.actions[index]

    def __len__(self):
        return len(self.actions)


def forward(i, states, target_actions, enc, dec, d_module, args,
            d_init=None, dec_mask=None):
    if args.framework == "mazebase":
        # cx_d = Variable(torch.zeros(states.size(0), args.lstm_dim))
        # hx_d = Variable(torch.zeros(states.size(0), args.lstm_dim))
        hx_d, cx_d = d_init(Variable(states[:, 0, :]).contiguous().cuda())
    else:
        cx_d = Variable(torch.zeros(states.size(0), args.dim))
        hx_d = Variable(torch.zeros(states.size(0), args.dim))

    if args.gpu:
        cx_d = cx_d.cuda()
        hx_d = hx_d.cuda()


    dec_loss = 0
    inv_loss =  0
    model_loss = 0
    recon_loss =  0
    forward_loss = 0


    current_epoch_actions = 0
    current_epoch_predicted_a_hat = 0

    s = None
    for r in range(args.rollout):
        curr_state = states[:, r, :]
        next_state = states[:, r + 1, :]
        if args.framework == "mazebase":
            curr_actions = Variable(target_actions[:, r].contiguous().view(
                -1, 1))
        else:
            curr_actions = Variable(target_actions[:, r].contiguous().view(
                -1, args.action_space.shape[0]))
        if s is None:
            s = Variable(curr_state.contiguous())
            if args.gpu:
                s = s.cuda()
            z = enc(s)
        s_prime = Variable(next_state.contiguous())
        if args.gpu:
            s_prime = s_prime.cuda()
        z_prime = enc(s_prime)

        if args.gpu:
            curr_actions = curr_actions.cuda()

        if args.framework == "mazebase":
            s_hat, s_hat_binary = dec(z)
            z_prime_hat, a_hat, (hx_d, cx_d) = d_module(
                z, curr_actions.long(), z_prime.detach(), (hx_d, cx_d))
            s_prime_hat, s_prime_hat_binary = dec(z_prime_hat)
            r_loss, m_loss, d_loss, i_loss, f_loss = get_maze_dynamics_losses(
                s, s_hat, s_prime, s_prime_hat, z_prime, z_prime_hat, a_hat,
                curr_actions, discrete=args.discrete, dec_mask= dec_mask)

            # caculate the accuracy here
            _, predicted_a = torch.max(F.sigmoid(a_hat),1)
            current_epoch_predicted_a_hat += (predicted_a == curr_actions.view(-1).long()).sum().data[0]
            current_epoch_actions += curr_actions.size(0)

        else:
            s_hat = dec(z)
            z_prime_hat, a_hat, (hx_d, cx_d) = d_module(
                (z, z_prime, curr_actions, (hx_d, cx_d)))
            s_prime_hat = dec(z_prime_hat)
            r_loss, m_loss, d_loss, i_loss, f_loss = get_dynamics_losses(
                s, s_hat, s_prime, s_prime_hat, z_prime, z_prime_hat,
                a_hat, curr_actions, discrete=args.discrete)

        inv_loss += i_loss
        dec_loss += d_loss
        forward_loss += f_loss
        recon_loss += r_loss
        model_loss += m_loss

        s = s_prime
        z = z_prime

    return forward_loss, inv_loss, dec_loss, recon_loss, model_loss, \
            current_epoch_predicted_a_hat, current_epoch_actions


def forward_planning(i, states, target_actions, enc, dec, d_module, args,
            d_init=None, dec_mask=None):
    cx_d = Variable(torch.zeros(states.size(0), args.dim))
    hx_d = Variable(torch.zeros(states.size(0), args.dim))

    if args.gpu:
        cx_d = cx_d.cuda()
        hx_d = hx_d.cuda()


    dec_loss = 0
    inv_loss =  0
    model_loss = 0
    recon_loss =  0
    forward_loss = 0


    current_epoch_actions = 0
    current_epoch_predicted_a_hat = 0

    s = None
    for r in range(args.rollout):
        curr_state = states[:, r, :]
        next_state = states[:, r + 1, :]
        curr_actions = Variable(target_actions[:, r].contiguous().view(
                -1, args.action_space.shape[0]))
        if s is None:
            s = Variable(curr_state.contiguous())
            if args.gpu:
                s = s.cuda()
            z = enc(s)
        s_prime = Variable(next_state.contiguous())
        if args.gpu:
            s_prime = s_prime.cuda()
        z_prime = enc(s_prime)

        if args.gpu:
            curr_actions = curr_actions.cuda()

        s_hat = dec(z)
        z_prime_hat, a_hat, (hx_d, cx_d) = d_module(
            (z, z_prime, curr_actions, (hx_d, cx_d)))
        s_prime_hat = dec(z_prime_hat)
        r_loss, m_loss, d_loss, i_loss, f_loss = get_dynamics_losses(
            s, s_hat, s_prime, s_prime_hat, z_prime, z_prime_hat,
            a_hat, curr_actions, discrete=args.discrete)

        inv_loss += i_loss
        dec_loss += d_loss
        forward_loss += f_loss
        recon_loss += r_loss
        model_loss += m_loss

        s = s_prime
        z = z_prime_hat

    return forward_loss, inv_loss, dec_loss, recon_loss, model_loss, \
            current_epoch_predicted_a_hat, current_epoch_actions


def multiple_forward(i, states, target_actions, enc, dec, d_module, args,
            d_init=None, dec_mask = None):
    cx_d = Variable(torch.zeros(states.size(0), args.dim))
    hx_d = Variable(torch.zeros(states.size(0), args.dim))

    if args.gpu:
        cx_d = cx_d.cuda()
        hx_d = hx_d.cuda()


    dec_loss = 0
    inv_loss =  0
    model_loss = 0
    recon_loss =  0
    forward_loss = 0


    current_epoch_actions = 0
    current_epoch_predicted_a_hat = 0

    s = None
    for r in range(args.rollout):
        curr_state = states[:, r, :]
        next_state = states[:, r + 1, :]
        if args.framework == "mazebase":
            curr_actions = Variable(target_actions[:, r].contiguous().view(
                -1, 1))
        else:
            curr_actions = Variable(target_actions[:, r].contiguous().view(
                -1, args.action_space.shape[0]))
        if s is None:
            s = Variable(curr_state.contiguous())
            if args.gpu:
                s = s.cuda()
            z = enc(s)
        s_prime = Variable(next_state.contiguous())
        if args.gpu:
            s_prime = s_prime.cuda()
        z_prime = enc(s_prime)

        if args.gpu:
            curr_actions = curr_actions.cuda()

        if args.framework == "mazebase":
            s_hat, s_hat_binary = dec(z)
            z_prime_hat, a_hat, (hx_d, cx_d) = d_module(
                z, curr_actions.long(), z_prime.detach(), (hx_d, cx_d))
            s_prime_hat, s_prime_hat_binary = dec(z_prime_hat)
            r_loss, m_loss, d_loss, i_loss, f_loss = get_maze_dynamics_losses(
                s, s_hat, s_prime, s_prime_hat, z_prime, z_prime_hat, a_hat,
                curr_actions, discrete=args.discrete, dec_mask= dec_mask)

            # caculate the accuracy here
            _, predicted_a = torch.max(F.sigmoid(a_hat),1)
            current_epoch_predicted_a_hat += (predicted_a == curr_actions.view(-1).long()).sum().data[0]
            current_epoch_actions += curr_actions.size(0)

        else:
            s_hat = dec(z)
            z_prime_hat, a_hat, (hx_d, cx_d) = d_module(
                (z, z_prime, curr_actions, (hx_d, cx_d)))
            s_prime_hat = dec(z_prime_hat)
            r_loss, m_loss, d_loss, i_loss, f_loss = get_dynamics_losses(
                s, s_hat, s_prime, s_prime_hat, z_prime, z_prime_hat, a_hat,
                curr_actions, discrete=args.discrete)

        inv_loss += i_loss
        dec_loss += d_loss
        forward_loss += f_loss
        recon_loss += r_loss
        model_loss += m_loss

        s = s_prime
        z = z_prime_hat

    return forward_loss, inv_loss, dec_loss, recon_loss, model_loss, \
                current_epoch_predicted_a_hat, current_epoch_actions

def train_dynamics(env, args, writer=None):
    """
    Trains the Dynamics module. Supervised.

    Arguments:
    env: the initialized environment (rllab/gym)
    args: input arguments
    writer: initialized summary writer for tensorboard
    """
    args.action_space = env.action_space

    # Initialize models
    enc = Encoder(env.observation_space.shape[0], args.dim,
                  use_conv=args.use_conv)
    dec = Decoder(env.observation_space.shape[0], args.dim,
                  use_conv=args.use_conv)
    d_module = D_Module(env.action_space.shape[0], args.dim, args.discrete)

    if args.from_checkpoint is not None:
        results_dict = torch.load(args.from_checkpoint)
        enc.load_state_dict(results_dict['enc'])
        dec.load_state_dict(results_dict['dec'])
        d_module.load_state_dict(results_dict['d_module'])

    all_params = chain(enc.parameters(), dec.parameters(), d_module.parameters())

    if args.transfer:
        for p in enc.parameters():
            p.requires_grad = False

        for p in dec.parameters():
            p.requires_grad = False
        all_params = d_module.parameters()

    optimizer = torch.optim.Adam(all_params, lr=args.lr,
                                 weight_decay=args.weight_decay)

    if args.gpu:
        enc = enc.cuda()
        dec = dec.cuda()
        d_module = d_module.cuda()

    # Initialize datasets
    val_loader = None
    train_dataset = DynamicsDataset(
        args.train_set, args.train_size, batch=args.train_batch,
        rollout=args.rollout)
    val_dataset = DynamicsDataset(args.test_set, 5000, batch=args.test_batch,
        rollout=args.rollout)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers)

    results_dict = {
        'dec_losses': [],
        'forward_losses': [],
        'inverse_losses': [],
        'total_losses': [],
        'enc': None,
        'dec': None,
        'd_module': None,
        'd_init':None,
        'args': args
    }

    total_action_taken = 0
    correct_predicted_a_hat = 0

    # create the mask here for re-weighting
    dec_mask = None
    if args.dec_mask is not None:
        dec_mask = torch.ones(9)
        game_vocab = dict([(b, a) for a, b in enumerate(sorted(env.game.all_possible_features()))])
        dec_mask[game_vocab['Agent']] = args.dec_mask
        dec_mask[game_vocab['Goal']] = args.dec_mask
        dec_mask =  dec_mask.expand(args.batch_size, args.maze_length,args.maze_length,9).contiguous().view(-1)
        dec_mask = Variable(dec_mask, requires_grad = False)
        if args.gpu:
            dec_mask = dec_mask.cuda()

    for epoch in range(1, args.num_epochs + 1):
        enc.train()
        dec.train()
        d_module.train()

        if args.framework == "mazebase":
            d_init.train()

        # for measuring the accuracy
        train_acc = 0
        current_epoch_actions = 0
        current_epoch_predicted_a_hat = 0

        start = time.time()
        for i, (states, target_actions) in enumerate(train_loader):

            optimizer.zero_grad()

            if args.framework != "mazebase":
                forward_loss, inv_loss, dec_loss, recon_loss, model_loss, _, _ = forward_planning(
                    i, states, target_actions, enc, dec, d_module, args)
            else:
                forward_loss, inv_loss, dec_loss, recon_loss, model_loss, current_epoch_predicted_a_hat, current_epoch_actions = multiple_forward(
                    i, states, target_actions, enc, dec, d_module, args, d_init, dec_mask )

            loss = forward_loss + args.inv_loss_coef * inv_loss + \
                        args.dec_loss_coef * dec_loss


            if i % args.log_interval == 0:
                log(
                    'Epoch [{}/{}]\tIter [{}/{}]\t'.format(
                        epoch, args.num_epochs, i+1, len(
                        train_dataset)//args.batch_size) + \
                    'Time: {:.2f}\t'.format(time.time() - start) + \
                    'Decoder Loss: {:.2f}\t'.format(dec_loss.data[0]) + \
                    'Forward Loss: {:.2f}\t'.format(forward_loss.data[0] ) + \
                    'Inverse Loss: {:.2f}\t'.format(inv_loss.data[0]) + \
                    'Loss: {:.2f}\t'.format(loss.data[0]))

                results_dict['dec_losses'].append(dec_loss.data[0])
                results_dict['forward_losses'].append(forward_loss.data[0])
                results_dict['inverse_losses'].append(inv_loss.data[0])
                results_dict['total_losses'].append(loss.data[0])

                # write the summaries here
                if writer:
                    writer.add_scalar('dynamics/total_loss', loss.data[0], epoch)
                    writer.add_scalar('dynamics/decoder', dec_loss.data[0], epoch)
                    writer.add_scalar(
                        'dynamics/reconstruction_loss', recon_loss.data[0], epoch)
                    writer.add_scalar(
                        'dynamics/next_state_prediction_loss',
                        model_loss.data[0], epoch)
                    writer.add_scalar('dynamics/inv_loss', inv_loss.data[0], epoch)
                    writer.add_scalar(
                        'dynamics/forward_loss', forward_loss.data[0], epoch)

                    writer.add_scalars(
                        'dynamics/all_losses',
                        {"total_loss":loss.data[0],
                            "reconstruction_loss":recon_loss.data[0],
                            "next_state_prediction_loss":model_loss.data[0],
                            "decoder_loss":dec_loss.data[0],
                            "inv_loss":inv_loss.data[0],
                            "forward_loss":forward_loss.data[0],
                        } , epoch)

            loss.backward()

            correct_predicted_a_hat += current_epoch_predicted_a_hat
            total_action_taken += current_epoch_actions

            # does it not work at all without grad clipping ?
            torch.nn.utils.clip_grad_norm(all_params, args.max_grad_norm)
            optimizer.step()

            # maybe add the generated image to add the logs
            # writer.add_image()

        # Run validation
        if val_loader is not None:
            enc.eval()
            dec.eval()
            d_module.eval()
            forward_loss, inv_loss, dec_loss = 0, 0, 0
            for i, (states, target_actions) in enumerate(val_loader):
                f_loss, i_loss, d_loss, _, _, _, _ = forward_planning(
                    i, states, target_actions, enc, dec, d_module, args)
                forward_loss += f_loss
                inv_loss += i_loss
                dec_loss += d_loss
            loss = forward_loss + args.inv_loss_coef * inv_loss + \
                    args.dec_loss_coef * dec_loss
            if writer:
                writer.add_scalar('val/forward_loss', forward_loss.data[0] / i, epoch)
                writer.add_scalar('val/inverse_loss', inv_loss.data[0] / i, epoch)
                writer.add_scalar('val/decoder_loss', dec_loss.data[0] / i, epoch)
            log(
                '[Validation]\t' + \
                'Decoder Loss: {:.2f}\t'.format(dec_loss.data[0] / i) + \
                'Forward Loss: {:.2f}\t'.format(forward_loss.data[0] / i) + \
                'Inverse Loss: {:.2f}\t'.format(inv_loss.data[0] / i) + \
                'Loss: {:.2f}\t'.format(loss.data[0] / i))
        if epoch % args.checkpoint == 0:
            results_dict['enc'] = enc.state_dict()
            results_dict['dec'] = dec.state_dict()
            results_dict['d_module'] = d_module.state_dict()
            if args.framework == "mazebase":
                results_dict['d_init'] = d_init.state_dict()
            torch.save(results_dict,
                os.path.join(args.out, 'dynamics_module_epoch%s.pt' % epoch))
            log('Saved model %s' % epoch)

    results_dict['enc'] = enc.state_dict()
    results_dict['dec'] = dec.state_dict()
    results_dict['d_module'] = d_module.state_dict()
    torch.save(results_dict,
               os.path.join(args.out, 'dynamics_module_epoch%s.pt' % epoch))
    print(os.path.join(args.out, 'dynamics_module_epoch%s.pt' % epoch))
