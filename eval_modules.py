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
from model import Encoder, Decoder, D_Module, R_Module
from common import *
from tensorboardX import SummaryWriter
from itertools import chain
from eval import test


def eval_reward(args, shared_model, writer_dir=None):
	"""
	For evaluation

	Arguments:
	- writer: the tensorboard summary writer directory (note: can't get it working directly with the SummaryWriter object)
	"""
	writer = SummaryWriter(log_dir=os.path.join(writer_dir,'eval')) if  writer_dir is not None else None

	# current episode stats
	episode_reward = episode_value_mse = episode_td_error = episode_pg_loss = episode_length = 0

	# global stats
	i_episode = 0
	total_episode = total_steps = 0
	num_goals_achieved = 0

	# intilialize the env and models
	torch.manual_seed(args.seed)
	env = create_env(args.env_name, framework=args.framework, args=args)
	set_seed(args.seed , env, args.framework)

	shared_enc, shared_dec, shared_d_module, shared_r_module = shared_model

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

	if args.from_checkpoint is not None:
		model_state, _ = torch.load(args.from_checkpoint)
		model.load_state_dict(model_state)

	# set the model to evaluation mode
	enc.eval()
	dec.eval()
	d_module.eval()
	r_module.eval()

	# reset the state
	state = env.reset()
	state = Variable(torch.from_numpy(state).float())

	start = time.time()

	while total_episode < args.num_episodes:

		# Sync with the shared model
		r_module.load_state_dict(shared_r_module.state_dict())
		d_module.load_state_dict(shared_d_module.state_dict())
		enc.load_state_dict(shared_enc.state_dict())
		dec.load_state_dict(shared_dec.state_dict())

		# reset stuff
		cd_p = Variable(torch.zeros(1, args.lstm_dim))
		hd_p = Variable(torch.zeros(1, args.lstm_dim))

		# for the reward
		cr_p = Variable(torch.zeros(1, args.lstm_dim))
		hr_p = Variable(torch.zeros(1, args.lstm_dim))

		i_episode += 1
		episode_length = 0
		episode_reward = 0
		args.local = True
		args.d = 0
		succ, _, episode_reward, episode_length = test(
			1, args, args, args, d_module, r_module, enc)
		log("Eval: succ {:.2f}, reward {:.2f}, length {:.2f}".format(
			succ, episode_reward, episode_length))
		# Episode has ended, write the summaries here
		if writer_dir is not None:
			# current episode stats
			writer.add_scalar('eval/episode_reward', episode_reward, i_episode)
			writer.add_scalar('eval/episode_length', episode_length, i_episode)
			writer.add_scalar('eval/success', succ, i_episode)

		time.sleep(args.eval_every)
		print("sleep")
