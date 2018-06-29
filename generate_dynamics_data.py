# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os

parser = argparse.ArgumentParser(description='Generate Data')
parser.add_argument('--env-name', default='InvertedPendulum-v1',
                    help='environment to train on (default: InvertedPendulum-v1)')
parser.add_argument('--N', type=int, default=1000000)
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--out', type=str, default='/data/ddr')
parser.add_argument('--num-processes', type=int, default=40,
                    help='how many training processes to use (default: 40)')
parser.add_argument('--rollout', type=int, default=20, help="rollout for goal")
parser.add_argument('--method', type=str, default='random',
                    help='["random", "pixel_control"]')
parser.add_argument('--render', action='store_true')
parser.add_argument('--reset', action='store_true')
parser.add_argument('--from-policy', type=str, default=None,
                    help="use reward module as policy")
parser.add_argument('--framework', default='gym',
                    help='framework of env (default: gym)')
parser.add_argument('--maze-id', type=int, default=0)
parser.add_argument('--maze-length', type=int, default=1)
parser.add_argument('--single-env', action='store_true')
parser.add_argument('--random-start', action='store_true')
parser.add_argument('-v', action='store_true', help='verbose logging')
parser.add_argument('--max-episode-length', type=int, default=500,
                    help='maximum length of an episode (default: 500)')
parser.add_argument('--file-path', type=str, default=None,
                    help='path to XML file for mujoco')


def generate_data(rank, args, start, end):

    from envs import create_env, set_seed, get_obs
    from model import R_Module
    import torch

    print(rank, "started")

    env = create_env(args.env_name, framework=args.framework, args=args)
    env = set_seed(args.seed + rank, env, args.framework)
    state = get_obs(env, args.framework)

    if args.from_policy is not None:
        model_state, r_args = torch.load(args.from_policy)
        policy = R_Module(env.action_space.shape[0],
                          r_args.dim,
    				   	  discrete=r_args.discrete, baseline=r_args.baseline,
    					  state_space=env.observation_space.shape[0])
        policy.load_state_dict(model_state)
        policy.eval()


    states = []
    actions = []
    i = start

    done = False

    while i < end:
        if i % 100 == 0:
            print(rank, i)
        ep_states = []
        ep_actions = []
        if args.from_policy is not None:
            cx_p = Variable(torch.zeros(1, r_args.dim))
            hx_p = Variable(torch.zeros(1, r_args.dim))
        for j in range(args.rollout):
            if args.from_policy is not None:
                value, logit, (hx_p, cx_p) = policy(
                    state.unsqueeze(0), (hx_p, cx_p))
                a, _, _ = get_action(logit, r_args.discrete)
            else:
                a = env.action_space.sample()
            ep_actions.append(a)

            state = get_obs(env, args.framework)
            env.step(a)

            if args.render:
                env.render()

            ep_states.append(state)

        final_state = get_obs(env, args.framework)
        ep_states.append(final_state)
        states.append(ep_states)
        actions.append(ep_actions)
        i += 1

        # reset the environment here
        if done or args.reset:
            env.reset()
            done = False

    torch.save((states, actions), os.path.join(
        args.out_dir, 'states_actions_%s_%s.pt' % (start, end)))



if __name__ == '__main__':
    import torch
    import torch.multiprocessing as mp
    mp.set_start_method('spawn')

    from torch.autograd import Variable
    from envs import create_env, set_seed, get_obs
    from model import R_Module
    os.environ['OMP_NUM_THREADS'] = '1'

    args = parser.parse_args()
    env_name = args.env_name
    env_name += '_rollout%s' % args.rollout
    if args.env_name.endswith('MazeEnv'):
        env_name += 'mazeid%slength%s' % (args.maze_id, args.maze_length)
        if args.single_env and args.maze_id == -1:
            env = create_env(args.env_name, framework=args.framework, args=args)
            env_name += '_single_env'
            args.maze_structure = env._env.MAZE_STRUCTURE
        if args.random_start:
            env_name += '_randomstart'
    if args.file_path is not None:
        env_name += '_transfer'
    if args.framework == 'mazebase':
        env_name += '_rollout_%s_length_%s' % (args.rollout, args.maze_length)
    args.out_dir = os.path.join(args.out, env_name)
    print(args)
    print(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    processes = []
    block = int(args.N / args.num_processes)
    for rank in range(0, args.num_processes):
        start = rank * block
        end = (rank + 1) * block
        p = mp.Process(target=generate_data, args=(rank, args, start, end))
        p.start()
        processes.append(p)

    torch.save(args, os.path.join(args.out_dir, 'args.pt'))

    # exit cleanly
    for p in processes:
        p.join()
