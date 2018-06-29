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

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from envs import *
from model import R_Module
from common import *
from tensorboardX import SummaryWriter

def ensure_shared_grads(model, shared_model):
	for param, shared_param in zip(model.parameters(),
								   shared_model.parameters()):
		if shared_param.grad is not None:
			return
		shared_param._grad = param.grad


def train_rewards(rank, args, shared_model, enc, optimizer=None, writer_dir=None,
				  d_module=None):
	"""

	Arguments:
	- writer: the tensorboard summary writer directory (note: can't get it working directly with the SummaryWriter object)
	"""
	# create writer here itself
	writer = None
	if writer_dir is not None:
		writer = SummaryWriter(log_dir=writer_dir)

	results_dict = {
		'reward': [],
		'policy_loss': [],
		'value_loss': [],
		'mean_entropy': [],
		'mean_predicted_value': []
	}

	running_t, running_reward, running_value_loss, running_policy_loss, \
		running_reward_loss = 0, 0, 0, 0, 0

	torch.manual_seed(args.seed + rank)
	env = create_env(args.env_name, framework=args.framework, args=args)
	set_seed(args.seed + rank, env, args.framework)
	model = R_Module(env.action_space.shape[0], args.dim,
					 discrete=args.discrete, baseline=args.baseline,
					 state_space=env.observation_space.shape[0])
	max_rollout = 0
	if args.planning:
		max_rollout = args.rollout

	if args.from_checkpoint is not None:
		model_state, _ = torch.load(args.from_checkpoint, map_location=lambda storage, loc: storage)
		model.load_state_dict(model_state)

	# no shared adam ?
	if optimizer is None:
		optimizer = optim.Adam(shared_model.parameters(), lr=args.lr, eps=args.eps)

	model.train()

	done = True
	episode_length = 0
	i_episode, total_episode = 0, 0
	start = time.time()
	while total_episode < args.num_episodes:
		# Sync with the shared model
		model.load_state_dict(shared_model.state_dict())
		if done:
			cx_p = Variable(torch.zeros(1, args.dim))
			hx_p = Variable(torch.zeros(1, args.dim))
			cx_d = Variable(torch.zeros(1, args.dim))
			hx_d = Variable(torch.zeros(1, args.dim))
			i_episode += 1
			episode_length = 0
			total_episode = args.num_steps * (i_episode - 1) + rank
			start = time.time()
			last_episode_length = episode_length
			if not args.single_env and args.env_name.endswith('MazeEnv'):  # generate new maze
				env = create_env(
					args.env_name, framework=args.framework, args=args)
			state = env.reset()
			state = Variable(torch.from_numpy(state).float())
			if not args.baseline:
				state = enc(state)
		else:
			cx_p = Variable(cx_p.data)
			hx_p = Variable(hx_p.data)
			cx_d = Variable(cx_d.data)
			hx_d = Variable(hx_d.data)

		values = []
		value_preds = []
		log_probs = []
		rewards = []
		total_actions = []
		entropies = []
		obses = []
		hx_ps = []
		cx_ps = []
		step = 0
		while step < args.num_steps:
			episode_length += 1
			if args.planning:
				_, actions, (hx_p, cx_p), (hx_d, cx_d), values, es, \
					lps = mcts(
					env, state, model, d_module, enc, (hx_p, cx_p), (hx_d, cx_d),
					args, discrete=args.discrete)
				log_probs += lps
				entropies += es
				actions = actions[:1]
			else:
				obses.append(state.unsqueeze(0))
				hx_ps.append(hx_p)
				cx_ps.append(cx_p)
				value, logit, (hx_p, cx_p) = model((
					state.unsqueeze(0), (hx_p, cx_p)))
				action, entropy, log_prob = get_action(
					logit, discrete=args.discrete)
				vlog("Action: %s\t Bounds: %s" % (str(action), str(
					(env.action_space.low, env.action_space.high))), args.v)
				entropies.append(entropy.mean().data)
				actions = [action]
				values.append(value)
				log_probs.append(log_prob)
			for action in actions:
				state, reward, done, _ = env.step(action.data.numpy())
				if args.neg_reward:
					reward = -reward
				state = Variable(torch.from_numpy(state).float())
				if args.clip_reward:
					reward = max(min(reward, 1), -1)
				if not args.baseline:
					state = enc(state)
				rewards.append(reward)
				total_actions.append(action)
				step += 1
				if done:
					break
			if done:
				break
		R = torch.zeros(1, 1)
		if not done:
			value, _, _ = model((state.unsqueeze(0), (hx_p, cx_p)))
			R = value.data
		done = True

		values.append(Variable(R))
		policy_loss = 0
		value_loss = 0
		advantages = np.zeros_like(rewards, dtype=float)
		R = Variable(R)
		gae = torch.zeros(1, 1)
		Rs = np.zeros_like(rewards, dtype=float)
		vlog("values: %s" % str([v.data[0,0] for v in values]), args.v)
		for i in reversed(range(len(rewards))):
			R = args.gamma * R + rewards[i]
			Rs[i] = R
			advantage = R - values[i]
			advantages[i] = advantage
			if args.algo == 'a3c':
				value_loss += 0.5 * advantage.pow(2)
				# Generalized Advantage Estimation
				if args.gae:
					delta_t = rewards[i] + args.gamma * values[i + 1].data \
						- values[i].data
					gae = gae * args.gamma * args.tau + delta_t
					policy_loss -= (log_probs[i] * Variable(gae).expand_as(
						log_probs[i])).mean()
				else:
					policy_loss -= advantage * (log_probs[i].mean())
		if args.algo == 'a3c':
			optimizer.zero_grad()
			(policy_loss + args.value_loss_coef * value_loss - \
				 args.entropy_coef * np.mean(entropies)).backward()
			torch.nn.utils.clip_grad_norm(model.parameters(), args.max_grad_norm)
			ensure_shared_grads(model, shared_model)
			optimizer.step()

		########Bookkeeping and logging#############
		U = 1. / min(i_episode, 100)
		running_reward = running_reward * (1 - U) + sum(rewards) * U
		running_t = running_t * (1 - U) + episode_length * U
		running_policy_loss = running_policy_loss * (1 - U) + policy_loss.squeeze().data[0] * U
		running_value_loss = running_value_loss * (1 - U) + \
			args.value_loss_coef * value_loss.squeeze().data[0] * U
		mean_entropy = np.mean([e.mean().data[0] for e in entropies])

		mean_predicted_value = np.mean([v.sum().data[0] for v in values])
		if total_episode % args.log_interval == 0 and done:
			if not args.discrete:
				sample_logits = (list(logit[0].data[0].numpy()),
								 list(logit[1].data[0].numpy()))
			else:
				sample_logits = list(logit.data[0].numpy())
			log(
				'Frames {}\t'.format(total_episode) + \
				'Avg reward: {:.2f}\tAverage length: {:.2f}\t'.format(
					running_reward, running_t) + \
				'Entropy: {:.2f}\tTime: {:.2f}\tRank: {}\t'.format(
					mean_entropy, time.time() - start, rank) + \
				'Policy Loss: {:.2f}\t'.format(running_policy_loss) + \
				# 'Reward Loss: {:.2f}\t'.format(running_reward_loss) + \
				'Weighted Value Loss: {:.2f}\t'.format(running_value_loss))
			vlog('Sample Action: %s\t' % str(list(action.data.numpy())) + \
				 'Logits: %s\t' % str(sample_logits), args.v)

			# write summaries here
			if writer_dir is not None and done:
				log('writing to tensorboard')
				# running losses
				writer.add_scalar('reward/running_reward', running_reward, i_episode)
				writer.add_scalar('reward/running_policy_loss', running_policy_loss, i_episode)
				writer.add_scalar('reward/running_value_loss', running_value_loss, i_episode)

				# current episode stats
				writer.add_scalar('reward/episode_reward', sum(rewards), i_episode)
				writer.add_scalar('reward/episode_policy_loss', policy_loss.squeeze().data[0], i_episode)
				writer.add_scalar('reward/episode_value_loss', value_loss.squeeze().data[0], i_episode)
				writer.add_scalar('reward/mean_entropy', mean_entropy, i_episode)
				writer.add_scalar('reward/mean_predicted_value', mean_predicted_value, i_episode)

				results_dict['reward'].append(sum(rewards))
				results_dict['policy_loss'].append(policy_loss.squeeze().data[0])
				results_dict['value_loss'].append(value_loss.squeeze().data[0])
				results_dict['mean_entropy'].append(mean_entropy)
				results_dict['mean_predicted_value'].append(mean_predicted_value)

		if total_episode % args.checkpoint_interval == 0:
			args.curr_iter = total_episode
			args.optimizer = optimizer
			torch.save((shared_model.state_dict(), args), os.path.join(
				args.out, args.model_name + '%s.pt' % total_episode))
			log("Saved model %d rank %s" % (total_episode, rank))
			log(os.path.join(
				args.out, args.model_name + '%s.pt' % total_episode))

		if writer_dir is not None and i_episode % \
				(args.checkpoint_interval // args.num_processes) == 0:
			torch.save(results_dict,
					   os.path.join(args.out, 'results_dict.pt'))
			log(os.path.join(args.out, 'results_dict.pt'))

	if writer_dir is not None:
		torch.save(results_dict,
			   os.path.join(args.out, 'results_dict.pt'))
		log(os.path.join(args.out, 'results_dict.pt'))
