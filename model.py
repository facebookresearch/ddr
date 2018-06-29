# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class Encoder(torch.nn.Module):
    def __init__(self, obs_space, dim, use_conv=False):
        """
        architecture should be input, so that we can pass multiple jobs !
        """
        super(Encoder, self).__init__()
        self.use_conv = use_conv
        self.obs_space = obs_space
        if use_conv:
            self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
            self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
            self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
            self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        else:
            self.linear1 = nn.Linear(obs_space, dim)
            self.linear2 = nn.Linear(dim, 32 * 3 * 3)
        self.fc = nn.Linear(32 * 3 * 3, dim)
        self.apply(weights_init)
        self.train()

    def forward(self, inputs):
        # why elu and not relu ?
        if self.use_conv:
            x = F.elu(self.conv1(inputs))
            x = F.elu(self.conv2(x))
            x = F.elu(self.conv3(x))
            x = F.elu(self.conv4(x))
        else:
            x = F.elu(self.linear1(inputs))
            x = F.elu(self.linear2(x))

        x = F.tanh(self.fc(x))

        return x


class Decoder(torch.nn.Module):
    def __init__(self, obs_space, dim, use_conv=False):
        super(Decoder, self).__init__()
        self.use_conv = use_conv
        self.fc = nn.Linear(dim, 32 * 3 * 3)
        if self.use_conv:
            self.deconv1 = nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1)
            self.deconv2 = nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1)
            self.deconv3 = nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1)
            self.deconv4 = nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1)
        else:
            self.linear1 = nn.Linear(32 * 3 * 3, dim)
            self.linear2 = nn.Linear(dim, obs_space)
        self.apply(weights_init)
        self.train()

    def forward(self, inputs):
        x = F.elu(self.fc(inputs))
        if self.use_conv:
            x = F.elu(self.deconv1(x))
            x = F.elu(self.deconv2(x))
            x = F.elu(self.deconv3(x))
            x = self.deconv4(x)
        else:
            x = F.elu(self.linear1(x))
            x = self.linear2(x)
        return x


class D_Module(torch.nn.Module):
    def __init__(self, action_space, dim, discrete=False):
        super(D_Module, self).__init__()
        self.dim = dim
        self.discrete = discrete

        self.za_embed = nn.Linear(2 * dim, dim)
        self.lstm_dynamics = nn.LSTMCell(dim, dim)
        self.z_embed = nn.Linear(dim, dim)

        self.inv = nn.Linear(2 * dim, dim)
        self.inv2 = nn.Linear(dim, action_space)

        self.action_linear = nn.Linear(action_space, dim)
        self.action_linear2 = nn.Linear(dim, dim)
        self.apply(weights_init)

        self.lstm_dynamics.bias_ih.data.fill_(0)
        self.lstm_dynamics.bias_hh.data.fill_(0)

        self.train()

    def forward(self, inputs):
        z, z_prime, actions, (hx_d, cx_d) = inputs
        z = z.view(-1, self.dim)

        a_embedding = F.elu(self.action_linear(actions))
        a_embedding = self.action_linear2(a_embedding)

        za_embedding = self.za_embed(
            torch.cat([z, a_embedding.view(z.size())], 1))
        hx_d, cx_d = self.lstm_dynamics(za_embedding, (hx_d, cx_d))
        z_prime_hat = F.tanh(self.z_embed(hx_d))

        # decode the action
        if z_prime is not None:
            z_prime = z_prime.view(-1, self.dim)
        else:
            z_prime = z_prime_hat
        a_hat = F.elu(self.inv(torch.cat([z, z_prime], 1)))
        a_hat = self.inv2(a_hat)
        return z_prime_hat, a_hat, (hx_d, cx_d)


class R_Module(torch.nn.Module):
    def __init__(self, action_space, dim, discrete=False, baseline=False,
                 state_space=None):
        super(R_Module, self).__init__()
        self.discrete = discrete
        self.baseline = baseline
        self.dim = dim

        if baseline:
            self.linear1 = nn.Linear(state_space, dim)
            self.linear2 = nn.Linear(dim, dim)
        self.lstm_policy = nn.LSTMCell(dim, dim)

        self.actor_linear = nn.Linear(dim, action_space)
        self.critic_linear = nn.Linear(dim, 1)
        self.rhat_linear = nn.Linear(dim, 1)
        if not discrete:
            self.actor_sigma_sq = nn.Linear(dim, action_space)

        self.apply(weights_init)

        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        # only forget should be 1
        self.lstm_policy.bias_ih.data.fill_(0)
        self.lstm_policy.bias_hh.data.fill_(0)

        if not discrete:
            self.actor_sigma_sq.weight.data = normalized_columns_initializer(
                self.actor_sigma_sq.weight.data, 0.01)
            self.actor_sigma_sq.bias.data.fill_(0)

        self.train()

    def forward(self, inputs):
        inputs, (hx_p, cx_p) = inputs
        if self.baseline:
            inputs = F.elu(self.linear1(inputs))
            inputs = F.elu(self.linear2(inputs))
        hx_p, cx_p = self.lstm_policy(inputs, (hx_p, cx_p))
        x = hx_p
        if self.discrete:
            action = self.actor_linear(x)
        else:
            action = (self.actor_linear(x), self.actor_sigma_sq(x))
        return self.critic_linear(x), action, (hx_p, cx_p)
