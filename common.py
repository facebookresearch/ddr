# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import math
import sys
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from model import Encoder, D_Module

pi = Variable(torch.FloatTensor([math.pi]))
def get_prob(x, mu, sigma_sq):
    a = (-1*(Variable(x)-mu).pow(2)/(2*sigma_sq + 1e-5)).exp()
    b = 1/(2*sigma_sq*pi.expand_as(sigma_sq) + 1e-5).sqrt()
    return a*b


def log(msg):
    print("[%s]\t%s" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg))
    sys.stdout.flush()


def vlog(msg, v):
    if v:
        log(msg)


def load_encoder(obs_space, args, freeze=True):
    enc = Encoder(obs_space, args.dim,
                  use_conv=args.use_conv)
    enc_state = torch.load(args.dynamics_module, map_location=lambda storage,
                           loc: storage)['enc']
    enc.load_state_dict(enc_state)
    enc.eval()
    if freeze:
        for p in enc.parameters():
            p.requires_grad = False
    return enc


def load_d_module(action_space, args, freeze=True):
    d_module_state = torch.load(args.dynamics_module, map_location=lambda storage,
                                loc: storage)['d_module']
    d_module = D_Module(action_space, args.dim, args.discrete)
    d_module.load_state_dict(d_module_state)
    d_module.eval()
    if freeze:
        for p in d_module.parameters():
            p.requires_grad = False
    return d_module


def get_action(logit, discrete, v=False):
    """Compute action, entropy, and log prob for discrete and continuous case
    from logit.
    """
    if discrete:
        prob = F.softmax(logit)
        log_prob = F.log_softmax(logit)
        # why entropy regularization ?
        entropy = -(log_prob * prob).sum(1, keepdim=True)
        action = prob.multinomial()
        log_prob = log_prob.gather(1, action)
    else:
        mu, sigma_sq = logit
        sigma_sq = F.softplus(sigma_sq)
        vlog('sigma_sq: %s' % str(sigma_sq.data), v)
        action = torch.normal(mu, sigma_sq)
        prob = get_prob(action.data, mu, sigma_sq) + 1e-5
        entropy = -0.5*((2 * sigma_sq * pi.expand_as(sigma_sq) + 1e-5).log() + 1)
        log_prob = prob.log()
    return action, entropy, log_prob


def eval_action(logit, action, discrete, v=False):
    mu, sigma_sq = logit
    sigma_sq = F.softplus(sigma_sq)
    vlog('sigma_sq: %s' % str(sigma_sq.data), v)
    prob = get_prob(action.data, mu, sigma_sq) + 1e-5
    entropy = -0.5*((2 * sigma_sq * pi.expand_as(sigma_sq) + 1e-5).log() + 1)
    log_prob = prob.log()
    return entropy, log_prob


def mcts(env, z_hat, r_module, d_module, enc, r_state, d_state, args, discrete,
		 use_env=False):
	import torch
	import torch.nn.functional as F
	from torch.autograd import Variable

	from common import get_action
	from envs import get_obs

	(hx_r, cx_r) = r_state
	(hx_d, cx_d) = d_state
	parent_states = [(z_hat, [], (hx_r, cx_r), (hx_d, cx_d), [], [], [])]
	child_states = []
	init_state = get_obs(env, args.framework)
	for i in range(args.d):
		actions = []
		best_val = None
		for z_hat, trajectory, (hx_r, cx_r), (hx_d, cx_d), val, entropies, \
				logprobs in parent_states:
			if best_val is None:
				best_val = val
			elif val < best_val:
				continue
			value, logit, (hx_r_prime, cx_r_prime) = r_module(
				(z_hat, (hx_r, cx_r)))
			val.append(value)
			if not discrete:
				for b in range(args.b):
					action, entropy, log_prob = get_action(
						logit, discrete=False, v=args.v)
					actions.append((action, entropy, log_prob))
			else:
				prob = F.softmax(logit)
				actions = np.argpartition(prob.data.numpy(), args.b)[:b]
			for a, e, lp in actions:
				if not use_env:
					z_prime_hat, _, (hx_d_prime, cx_d_prime) = d_module(
						(z_hat, z_hat, a, (hx_d, cx_d)))
				else:
					state = get_obs(env, args.framework)
					for t in trajectory:
						env.step(t.data.numpy())
					s_prime, _, _, _ = env.step(a.data.numpy())
					s_prime = Variable(torch.from_numpy(s_prime).float())
					z_prime_hat = enc(s_prime).unsqueeze(0)
					env.reset(state)
					hx_d_prime, cx_d_prime = hx_d, cx_d
				child_states.append(
					(z_prime_hat, trajectory + [a], (hx_r_prime, cx_r_prime),
					(hx_d_prime, cx_d_prime), val, entropies + [e], logprobs + [lp]))
		child_states = prune(child_states, b)
		parent_states = child_states
		child_states = []

	# compute value of final state in each trajectory and choose best
	best_val = sum(parent_states[0][4]).data[0,0]
	best_ind = 0
	for ind, (z, traj, hr, hd, v, _, _) in enumerate(parent_states):
		vr, _, _ = r_module((z, hr))
		v.append(vr)
		if sum(v).data[0,0] > best_val:
			best_ind = ind
			best_val = sum(v).data[0,0]
	return parent_states[best_ind]
