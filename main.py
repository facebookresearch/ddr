# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import print_function

import datetime
import os
import time
import shutil
from itertools import chain
import dill

from arguments import get_args


if __name__ == '__main__':
    import torch
    import torch.multiprocessing as mp
    mp.set_start_method('spawn')

    import my_optim
    from envs import create_env
    from model import *
    from test import test
    from train_reward_module import train_rewards
    from common import *
    from train_dynamics_module import train_dynamics
    from train_online import train_online
    from eval_modules import eval_reward
    from tensorboardX import SummaryWriter

    os.environ['OMP_NUM_THREADS'] = '1'
    args = get_args()
    log(args)

    if not args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ""

    torch.manual_seed(args.seed)

    args_param = vars(args)
    toprint = ['seed', 'lr', 'entropy_coef', 'value_loss_coef', 'num_steps',
               'dim']

    if args.planning:
        toprint += ['rollout']

    env_name = args.env_name
    if args.env_name.endswith("MazeEnv"):
        env_name += 'mazeid%slength%s' % (args.maze_id, args.maze_length)
        toprint += ['random_start', 'difficulty']
    if args.baseline:
        model_type = 'baseline'
        if args.neg_reward:
            model_type += '_neg_reward'
        if args.file_path:
            model_type += '_dynamics_transfer'
        toprint += ['algo', 'gae', 'num_processes']
    elif args.train_dynamics:
        model_type = 'dynamics_planning'
        toprint = ['lr', 'forward_loss_coef', 'dec_loss_coef', 'inv_loss_coef', 'rollout', 'dim',
                   'train_size']
        # env_name = os.path.basename(args.train_set.strip('/'))
        if args.single_env:
            data_args = torch.load(os.path.join(args.train_set, 'args.pt'))
            args.maze_structure = data_args.maze_structure
    elif args.train_reward:
        model_type = 'reward'
        if args.neg_reward:
            model_type += '_neg_reward'
        if args.file_path:
            model_type += '_dynamics_transfer'
        toprint += ['algo', 'gae']
        if args.planning:
            model_type += '_planning'
    elif args.train_online:
        model_type = 'online'
        toprint += ['lr', 'dec_loss_coef', 'inv_loss_coef', 'rollout', 'dim']
    if args.transfer:
        model_type += '_transfer'

    name = ''
    for arg in toprint:
        name += '_{}{}'.format(arg, args_param[arg])
    out_dir = os.path.join(args.out, env_name, model_type, name)
    args.out = out_dir

    dynamics_path = ''
    if args.dynamics_module is not None and not args.baseline:
        dynamics_path = args.dynamics_module.split('/')
        dynamics_path = dynamics_path[-4] + dynamics_path[-2] +\
            '_' + dynamics_path[-1].strip('.pt')
        args.out = os.path.join(out_dir, dynamics_path)
    os.makedirs(args.out, exist_ok=True)

    # create the tensorboard summary writer here
    tb_log_dir = os.path.join(args.log_dir, env_name, model_type, name,
        dynamics_path, 'tb_logs')
    print(tb_log_dir)
    print(args.out)

    if args.reset_dir:
        shutil.rmtree(tb_log_dir, ignore_errors=True)
    os.makedirs(tb_log_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=tb_log_dir)

    # dump all the arguments in the tb_log_dir
    print(args, file=open(os.path.join(tb_log_dir, "arguments"), "w"))


    env = create_env(args.env_name, framework=args.framework, args=args)
    if args.train_dynamics:
        train_dynamics(env, args, None) # tb_writer
    if args.train_reward:
        model_name = 'rewards_module'
        if args.from_checkpoint is not None:  # using curriculum
            model_name += 'curr'
        if args.single_env:
            model_name += '_single_env'
            args.maze_structure = env._env.MAZE_STRUCTURE
        args.model_name = model_name
        enc = None
        d_module = None
        assert args.dynamics_module is not None
        enc = load_encoder(env.observation_space.shape[0], args)
        if args.planning:
            d_module = load_d_module(env.action_space.shape[0], args)

        shared_model = R_Module(env.action_space.shape[0], args.dim,
                            discrete=args.discrete, baseline=args.baseline,
                            state_space=env.observation_space.shape[0])

        # shared reward module for everyone
        shared_model.share_memory()

        if args.no_shared:
            optimizer = None
        else:
            optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=args.lr)
            optimizer.share_memory()

        processes = []

        train_agent_method = None

        total_args = args
        train_agent_method = train_rewards

        for rank in range(0, args.num_processes):
            if rank==0:
                p = mp.Process(target=train_agent_method, args=(
                    rank, total_args, shared_model, enc, optimizer, tb_log_dir,
                    d_module))
            else:
                p = mp.Process(target=train_agent_method, args=(
                    rank, total_args, shared_model, enc, optimizer, None, d_module))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        torch.save((shared_model.state_dict(), args), os.path.join(
            args.out, model_name + '%s.pt' % args.num_episodes))

        print(os.path.join(args.out, model_name))
    if args.train_online:
        model_name = 'rewards_module'
        if args.from_checkpoint is not None:  # using curriculum
            model_name += 'curr'
        if args.single_env:
            model_name += '_single_env'
            args.maze_structure = env._env.MAZE_STRUCTURE
        args.model_name = model_name
        shared_enc = Encoder(env.observation_space.shape[0], args.dim,
                      use_conv=args.use_conv)
        shared_dec = Decoder(env.observation_space.shape[0], args.dim,
                      use_conv=args.use_conv)
        shared_d_module = D_Module(env.action_space.shape[0], args.dim,
                                   args.discrete)
        shared_r_module = R_Module(env.action_space.shape[0], args.dim,
                                discrete=args.discrete, baseline=args.baseline,
                                state_space=env.observation_space.shape[0])

        shared_enc = Encoder(env.observation_space.shape[0], args.dim,
                      use_conv=args.use_conv)
        shared_dec = Decoder(env.observation_space.shape[0], args.dim,
                      use_conv=args.use_conv)
        shared_d_module = D_Module(env.action_space.shape[0], args.dim,
                                   args.discrete)
        shared_r_module = R_Module(env.action_space.shape[0], args.dim,
                                discrete=args.discrete, baseline=args.baseline,
                                state_space=env.observation_space.shape[0])

        shared_enc.share_memory()
        shared_dec.share_memory()
        shared_d_module.share_memory()
        shared_r_module.share_memory()
        all_params = chain(shared_enc.parameters(), shared_dec.parameters(),
                           shared_d_module.parameters(),
                           shared_r_module.parameters())
        shared_model = [shared_enc, shared_dec, shared_d_module, shared_r_module]

        if args.single_env:
            model_name += '_single_env'
            args.maze_structure = env.MAZE_STRUCTURE

        if args.no_shared:
            optimizer = None
        else:
            optimizer = my_optim.SharedAdam(all_params, lr=args.lr)
            optimizer.share_memory()

        train_agent_method = train_online

        processes = []
        for rank in range(0, args.num_processes):
            if rank==0:
                p = mp.Process(target=train_agent_method, args=(
                    rank, args, shared_model, optimizer, tb_log_dir))
            else:
                p = mp.Process(target=train_agent_method, args=(
                    rank, args, shared_model, optimizer))
            p.start()
            processes.append(p)

        # start an eval process here
        eval_agent_method = eval_reward
        p = mp.Process(target=eval_agent_method, args=(
            args, shared_model, tb_log_dir))
        p.start()
        processes.append(p)

        for p in processes:
            p.join()
        results_dict = {'args': args}
        torch.save((shared_r_module.state_dict(), args), os.path.join(
            args.out, 'reward_module%s.pt' % args.num_episodes))
        results_dict['enc'] = shared_enc.state_dict()
        results_dict['dec'] = shared_dec.state_dict()
        results_dict['d_module'] = shared_d_module.state_dict()
        torch.save(results_dict,
            os.path.join(args.out, 'dynamics_module%s.pt' % args.num_episodes))
        log("Saved model %s" % os.path.join(args.out, model_name))
