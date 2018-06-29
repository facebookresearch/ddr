# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import print_function

import argparse
import numpy as np
import os
import random
from operator import itemgetter

# Environment settings
parser = argparse.ArgumentParser(description='Eval DDR')

parser.add_argument('--dynamics-module', type=str, default=None,
					help='Dynamics module')
parser.add_argument('--rewards-module', type=str, default=None,
					help='Rewards module')
parser.add_argument('--num-processes', type=int, default=20,
					help='how many training processes to use (default: 20)')
parser.add_argument('--N', type=int, default=1,
					help='Number of episodes')
parser.add_argument('--rollout', type=int, default=20, help="rollout for goal")
parser.add_argument('--seed', type=int, default=1,
					help='random seed (default: 1)')
parser.add_argument('--render', action='store_true')
parser.add_argument('--out', type=str, default=None)
parser.add_argument('--max-episode-length', type=int, default=1000,
					help='maximum length of an episode')
parser.add_argument('--framework', default='gym',
					help='framework of env (default: gym)')
parser.add_argument('--env-name', default='InvertedPendulum-v1',
					help='environment to train on (default: InvertedPendulum-v1)')
parser.add_argument('--maze-id', type=int, default=0)
parser.add_argument('--maze-length', type=int, default=1)
parser.add_argument('--log-interval', type=int, default=1)
parser.add_argument('--baseline', action='store_true')
parser.add_argument('--local', action='store_true',
					help='running locally to render, no multiprocessing')
parser.add_argument('--single-env', action='store_true')
parser.add_argument('--coef-inner-rew', type=float, default=1.)
parser.add_argument('--mcts', action='store_true', help='Monte Carlo Tree Search')
parser.add_argument('-b', type=int, default=4, help='branching factor')
parser.add_argument('-d', type=int, default=3, help='planning depth')
parser.add_argument('--file-path', type=str, default=None,
					help='path to XML file for mujoco')
parser.add_argument('--save-figs', action='store_true')
parser.add_argument('--neg-reward', action='store_true',
					help='set reward negative for transfer')
parser.add_argument('--use-env', action='store_true', help='Use env with MCTS')
parser.add_argument('-v', action='store_true', help='verbose logging')
parser.add_argument('--difficulty', type=int, default=-1, help='difficulty of maze')


def prune(states, b):
	"""Prune states down to length b, sorting by val."""
	return sorted(states, key=itemgetter(4))[:b]


def test(block, args, d_args, r_args, d_module, r_module, enc, dec, q=None, rank=0):
	import torch
	from torch.autograd import Variable

	from envs import create_env, reset_env, get_obs
	from common import get_action, log

	seed = args.seed * 9823 + 194885 + rank # make sure doesn't copy train
	torch.manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
	i = 1
	total_acc, total_reward = [], []
	avg_succ, avg_reward, avg_len = 0, 0, 0
	while len(total_acc) < block:
		reward_sum, succ = 0, 0
		actions = []
		if args.single_env and i > 1:
			reset_env(env, args)
		else:
			env = create_env(args.env_name, framework=args.framework, args=args, eval_flag=True)
		done = False
		step = 0

		# Should the two LSTMs share a hidden state?
		cx_r = Variable(torch.zeros(1, r_args.dim))
		hx_r = Variable(torch.zeros(1, r_args.dim))
		if not args.baseline:
			cx_d = Variable(torch.zeros(1, d_args.dim))
			hx_d = Variable(torch.zeros(1, d_args.dim))
		while step < args.max_episode_length and not done:
			# Encode state
			state = get_obs(env, r_args.framework)
			state = Variable(torch.from_numpy(state).float())
			if not args.baseline:
				z = enc(state)
				z_prime_hat = z.unsqueeze(0)
			else:
				z_prime_hat = state.unsqueeze(0)
			actions = []
			if args.mcts:
				z_prime_hat, actions, (hx_r, cx_r), (hx_d, cx_d), _, _, _ = mcts(
					env, z_prime_hat, r_module, d_module, enc, (hx_r, cx_r),
					(hx_d, cx_d), args, discrete=r_args.discrete,
					use_env=args.use_env)
			for r in range(args.rollout - args.d):
				value, logit, (hx_r, cx_r) = r_module(
					(z_prime_hat, (hx_r, cx_r)))
				action, entropy, log_prob = get_action(
					logit, discrete=r_args.discrete)
				actions.append(action)
				if not args.baseline:
					z_prime_hat, _, (hx_d, cx_d) = d_module(
						(z_prime_hat, z_prime_hat, action, (hx_d, cx_d)))
					if args.save_figs:
						s_prime_hat = dec(z_prime_hat)

			for action in actions[:args.rollout]:
				_, reward, done, _ = env.step(action.data.numpy())
				if args.render:
					env.render()
				reward_sum += reward
				step += 1
				if done:
					succ = 1
					break
		U = 1. / i
		total_acc.append(succ)
		total_reward.append(reward_sum)
		avg_succ = avg_succ * (1 - U) + succ * U
		avg_reward = avg_reward * (1 - U) + reward_sum * U
		avg_len = avg_len * (1 - U) + (step + 1) * U
		if i % args.log_interval == 0:
			log("Eval: {:d} episodes, avg succ {:.2f}, avg reward {:.2f}, avg length {:.2f}".format(
				len(total_acc), avg_succ, reward_sum, step))
		i += 1
	if args.local:
		return (sum(total_acc), len(total_acc), sum(total_reward), avg_len)
	q.put((sum(total_acc), len(total_acc), sum(total_reward)))


if __name__ == '__main__':
	import torch
	import torch.multiprocessing as mp
	mp.set_start_method('spawn')

	from envs import *
	from model import *
	from common import *
	# from ppo.model import MLPPolicy

	os.environ['OMP_NUM_THREADS'] = '1'
	os.environ['CUDA_VISIBLE_DEVICES'] = ""

	args = parser.parse_args()
	if not args.mcts:
		args.d = 0
	log(args)
	torch.manual_seed(args.seed)

	d_args, d_module, enc, dec = None, None, None, None
	r_state_dict, r_args = torch.load(args.rewards_module, map_location=lambda storage, loc: storage)
	if args.single_env and hasattr(r_args, 'maze_structure'):
		args.maze_structure = r_args.maze_structure
	env = create_env(args.env_name, framework=args.framework, args=args, eval_flag=True)
	r_module = R_Module(env.action_space.shape[0], r_args.dim,
					 discrete=r_args.discrete, baseline=r_args.baseline,
					 state_space=env.observation_space.shape[0])
	r_module.load_state_dict(r_state_dict)
	r_module.eval()
	if not args.baseline:
		if args.local:
			r_args.dynamics_module = '/Users/amyzhang/ddr_for_tl' + r_args.dynamics_module[24:]
		if args.dynamics_module is None:
			d_dict = torch.load(r_args.dynamics_module, map_location=lambda storage, loc: storage)
		else:
			d_dict = torch.load(args.dynamics_module, map_location=lambda storage, loc: storage)
		d_args = d_dict['args']
		enc_state = d_dict['enc']
		dec_state = d_dict['dec']
		d_state_dict = d_dict['d_module']
		d_module = D_Module(env.action_space.shape[0], d_args.dim, d_args.discrete)
		d_module.load_state_dict(d_state_dict)
		d_module.eval()

		enc = Encoder(env.observation_space.shape[0], d_args.dim,
				      use_conv=d_args.use_conv)
		dec = Decoder(env.observation_space.shape[0], d_args.dim,
                      use_conv=d_args.use_conv)
		enc.load_state_dict(enc_state)
		dec.load_state_dict(dec_state)
		enc.eval()
		dec.eval()

	block = int(args.N / args.num_processes)
	if args.local:
		all_succ, all_total, avg_reward = test(
			block, args, d_args, r_args, d_module, r_module, enc, dec)
	else:
		processes = []
		queues = []
		for rank in range(0, args.num_processes):
			q = mp.Queue()
			p = mp.Process(target=test, args=(
				block, args, d_args, r_args, d_module, r_module, enc, dec, q, rank))
			p.start()
			processes.append(p)
			queues.append(q)

		for i, p in enumerate(processes):
			log("Exit process %d" % i)
			p.join()

		all_succ = 0
		all_total = 0
		total_reward = 0
		for q in queues:
			while not q.empty():
				succ, total, total_r = q.get()
				all_succ += succ
				all_total += total
				total_reward += total_r
	log("Success: %s, %s, %s" % (all_succ / all_total, all_succ, all_total))
	log("Average Reward: %s" % (total_reward / all_total))
	if args.out:
		with open(args.out, 'a') as f:
			f.write("Success: %s \n" % (all_succ / all_total))
