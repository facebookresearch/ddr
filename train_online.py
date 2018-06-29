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
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from envs import *
from model import Encoder, Decoder, D_Module, R_Module
from train_dynamics_module import D_Module, get_dynamics_losses
from common import *
from tensorboardX import SummaryWriter

def ensure_shared_grads(model, shared_model):
	for param, shared_param in zip(model.parameters(),
								   shared_model.parameters()):
		if shared_param.grad is not None:
			return
		shared_param._grad = param.grad


def train_online(rank, args, shared_model, optimizer=None, writer_dir=None):
	"""

	Arguments:
	- writer: the tensorboard summary writer directory (note: can't get it working directly with the SummaryWriter object)
	"""
	# create writer here itself
	writer = None
	if writer_dir is not None:
		writer = SummaryWriter(log_dir=writer_dir)

	shared_enc, shared_dec, shared_d_module, shared_r_module = shared_model
	running_t, running_reward, running_value_loss, running_policy_loss, \
		running_reward_loss = 0, 0, 0, 0, 0

	torch.manual_seed(args.seed + rank)
	env = create_env(args.env_name, framework=args.framework, args=args)
	set_seed(args.seed + rank, env, args.framework)
	enc = Encoder(env.observation_space.shape[0], args.dim,
				  use_conv=args.use_conv)
	dec = Decoder(env.observation_space.shape[0], args.dim,
				  use_conv=args.use_conv)
	d_module = D_Module(env.action_space.shape[0], args.dim, args.discrete)
	r_module = R_Module(env.action_space.shape[0], args.dim,
					 discrete=args.discrete, baseline=False,
					 state_space=env.observation_space.shape[0])

	all_params = chain(enc.parameters(), dec.parameters(),
					   d_module.parameters(),
					   r_module.parameters())
	# no shared adam ?
	if optimizer is None:
		optimizer = optim.Adam(all_params, lr=args.lr)

	enc.train()
	dec.train()
	d_module.train()
	r_module.train()

	results_dict = {
		'enc': None,
		'dec': None,
		'd_module': None,
		'args': args,
		'reward': [],
		'policy_loss': [],
		'value_loss': [],
		'mean_entropy': [],
		'mean_predicted_value': [],
		'dec_losses': [],
		'forward_losses': [],
		'inverse_losses': [],
		'total_losses': [],
	}
	episode_length = 0
	i_episode, total_episode = 0, 0
	done = True
	start = time.time()
	while total_episode < args.num_episodes:
		# Sync with the shared model
		r_module.load_state_dict(shared_r_module.state_dict())
		d_module.load_state_dict(shared_d_module.state_dict())
		enc.load_state_dict(shared_enc.state_dict())
		dec.load_state_dict(shared_dec.state_dict())
		if done:
			cx_p = Variable(torch.zeros(1, args.dim))
			hx_p = Variable(torch.zeros(1, args.dim))
			cx_d = Variable(torch.zeros(1, args.dim))
			hx_d = Variable(torch.zeros(1, args.dim))
			i_episode += 1
			episode_length = 0
			total_episode = args.num_processes * (i_episode - 1) + rank
			start = time.time()
			last_episode_length = episode_length
			if not args.single_env and args.env_name.endswith('MazeEnv'):  # generate new maze
				env = create_env(
					args.env_name, framework=args.framework, args=args)
			s = env.reset()
			s = Variable(torch.from_numpy(s).float())
		else:
			cx_p = Variable(cx_p.data)
			hx_p = Variable(hx_p.data)
			cx_d = Variable(cx_d.data)
			hx_d = Variable(hx_d.data)
			s = Variable(s.data)
		z = enc(s).unsqueeze(0)
		s_hat = dec(z)

		values = []
		rhats = []
		log_probs = []
		rewards = []
		entropies = []
		dec_loss = 0
		inv_loss = 0
		model_loss = 0
		recon_loss = 0
		forward_loss = 0
		for step in range(args.num_steps):
			episode_length += 1
			value, rhat, logit, (hx_p, cx_p) = r_module((
				z.detach(), (hx_p, cx_p)))
			action, entropy, log_prob = get_action(logit, discrete=args.discrete)
			vlog("Action: %s\t Bounds: %s" % (str(action), str((env.action_space.low, env.action_space.high))), args.v)
			entropies.append(entropy)
			s_prime, reward, done, _ = env.step(action.data.numpy())
			s_prime = Variable(torch.from_numpy(s_prime).float())
			done = done or episode_length >= args.max_episode_length

			z_prime = enc(s_prime)
			z_prime_hat, a_hat, (hx_d, cx_d) = d_module(
				(z, z_prime, action, (hx_d, cx_d)))
			s_prime_hat = dec(z_prime_hat)
			r_loss, m_loss, d_loss, i_loss, f_loss = get_dynamics_losses(
				s, s_hat, s_prime, s_prime_hat, z_prime, z_prime_hat, a_hat,
				action)
			values.append(value)
			rhats.append(rhat)
			log_probs.append(log_prob)
			rewards.append(reward)
			dec_loss += d_loss
			inv_loss += i_loss
			model_loss += m_loss
			recon_loss += r_loss
			forward_loss += f_loss
			z = z_prime_hat
			s = s_prime
			s_hat = s_prime_hat
			if done:
				break
		R = torch.zeros(1, 1)
		if not done:
			value, _, _, _ = r_module((z, (hx_p, cx_p)))
			R = value.data

		values.append(Variable(R))
		policy_loss = 0
		value_loss = 0
		rew_loss = 0
		pred_reward_loss = 0
		R = Variable(R)
		gae = torch.zeros(1, 1)
		vlog("values: %s" % str([v.data[0,0] for v in values]), args.v)
		vlog("rhats: %s" % str(rhats), args.v)
		for i in reversed(range(len(rewards))):
			R = args.gamma * R + rewards[i]
			advantage = R - values[i]
			value_loss += 0.5 * advantage.pow(2)

			# reward loss
			rew_loss += F.mse_loss(rhats[i], Variable(torch.from_numpy(
				np.array([rewards[i]])).float()))

			# Generalized Advantage Estimation
			delta_t = rewards[i] + args.gamma * values[i + 1].data \
				- values[i].data
			gae = gae * args.gamma * args.tau + delta_t
			if args.discrete:
				policy_loss = policy_loss - log_probs[i] * Variable(gae) \
					- args.entropy_coef * entropies[i]
			else:
				policy_loss = policy_loss - (log_probs[i] * Variable(gae).expand_as(
					log_probs[i])).sum() - (args.entropy_coef * entropies[i]).sum()

		optimizer.zero_grad()
		U = 1. / min(i_episode, 100)
		running_reward = running_reward * (1 - U) + sum(rewards) * U
		running_t = running_t * (1 - U) + episode_length * U
		running_policy_loss = running_policy_loss * (1 - U) + policy_loss.data[0] * U
		running_value_loss = running_value_loss * (1 - U) + \
			args.value_loss_coef * value_loss.data[0, 0] * U
		running_reward_loss = running_reward_loss * (1 - U) + \
			args.rew_loss_coef * rew_loss.data[0] * U
		mean_entropy = np.mean([e.sum().data[0] for e in entropies])

		mean_predicted_value = np.mean([v.sum().data[0] for v in values])
		loss = policy_loss + args.value_loss_coef * value_loss + \
			args.rew_loss_coef * rew_loss + args.inv_loss_coef * inv_loss + \
			args.dec_loss_coef * dec_loss + forward_loss
		if total_episode % args.log_interval == 0 and done:
			if not args.discrete:
				sample_logits = (list(logit[0].data[0].numpy()),
								 list(logit[1].data[0].numpy()))
			else:
				sample_logits = list(logit.data[0].numpy())
			log(
				'Episode {}\t'.format(total_episode) + \
				'Avg reward: {:.2f}\tAverage length: {:.2f}\t'.format(
					running_reward, running_t) + \
				'Entropy: {:.2f}\tTime: {:.2f}\tRank: {}\t'.format(
					mean_entropy, time.time() - start, rank) + \
				'Policy Loss: {:.2f}\t'.format(running_policy_loss) + \
				'Reward Loss: {:.2f}\t'.format(running_reward_loss) + \
				'Weighted Value Loss: {:.2f}\t'.format(running_value_loss) + \
				'Sample Action: %s\t' % str(list(action.data.numpy())) + \
				'Logits: %s\t' % str(sample_logits) + \
				'Decoder Loss: {:.2f}\t'.format(dec_loss.data[0]) + \
				'Forward Loss: {:.2f}\t'.format(forward_loss.data[0]) + \
				'Inverse Loss: {:.2f}\t'.format(inv_loss.data[0]) + \
				'Loss: {:.2f}\t'.format(loss.data[0, 0]))

		# write summaries here
		if writer_dir is not None and done:
			log('writing to tensorboard')

			# running losses
			writer.add_scalar('reward/running_reward', running_reward, i_episode)
			writer.add_scalar('reward/running_policy_loss', running_policy_loss, i_episode)
			writer.add_scalar('reward/running_value_loss', running_value_loss, i_episode)

			# current episode stats
			writer.add_scalar('reward/episode_reward', sum(rewards), i_episode)
			writer.add_scalar('reward/episode_policy_loss', policy_loss.data[0], i_episode)
			writer.add_scalar('reward/episode_value_loss', value_loss.data[0,0], i_episode)
			writer.add_scalar('reward/mean_entropy', mean_entropy, i_episode)
			writer.add_scalar('reward/mean_predicted_value', mean_predicted_value, i_episode)
			writer.add_scalar('dynamics/total_loss', loss.data[0], i_episode)
			writer.add_scalar('dynamics/decoder', dec_loss.data[0], i_episode)
			writer.add_scalar('dynamics/reconstruction_loss', recon_loss.data[0], i_episode)
			writer.add_scalar('dynamics/next_state_prediction_loss', model_loss.data[0], i_episode)
			writer.add_scalar('dynamics/inv_loss', inv_loss.data[0], i_episode)
			writer.add_scalar('dynamics/forward_loss', forward_loss.data[0], i_episode)

			results_dict['reward'].append(sum(rewards))
			results_dict['policy_loss'].append(policy_loss.data[0])
			results_dict['value_loss'].append(value_loss.data[0,0])
			results_dict['mean_entropy'].append(mean_entropy)
			results_dict['mean_predicted_value'].append(mean_predicted_value)
			results_dict['dec_losses'].append(dec_loss.data[0])
			results_dict['forward_losses'].append(forward_loss.data[0])
			results_dict['inverse_losses'].append(inv_loss.data[0])
			results_dict['total_losses'].append(loss.data[0])

		loss.backward()
		torch.nn.utils.clip_grad_norm(all_params, args.max_grad_norm)
		ensure_shared_grads(r_module, shared_r_module)
		ensure_shared_grads(d_module, shared_d_module)
		ensure_shared_grads(enc, shared_enc)
		ensure_shared_grads(dec, shared_dec)
		optimizer.step()

		if total_episode % args.checkpoint_interval == 0:
			args.curr_iter = total_episode
			args.dynamics_module = os.path.join(
				args.out, 'dynamics_module%s.pt' % total_episode)
			torch.save((shared_r_module.state_dict(), args), os.path.join(
				args.out, 'reward_module%s.pt' % total_episode))
			results_dict['enc'] = shared_enc.state_dict()
			results_dict['dec'] = shared_dec.state_dict()
			results_dict['d_module'] = shared_d_module.state_dict()
			torch.save(results_dict,
				os.path.join(args.out, 'dynamics_module%s.pt' % total_episode))
			log("Saved model %d" % total_episode)

		if writer_dir is not None and i_episode % \
				(args.checkpoint_interval // args.num_processes) == 0:
			torch.save(results_dict,
					   os.path.join(args.out, 'results_dict.pt'))
			print(os.path.join(args.out, 'results_dict.pt'))

	if writer_dir is not None:
		torch.save(results_dict,
				   os.path.join(args.out, 'results_dict.pt'))
		print(os.path.join(args.out, 'results_dict.pt'))
