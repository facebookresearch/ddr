# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np

import gym
from gym.spaces.box import Box
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.mujoco.ant_env import AntEnv
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.mujoco.hopper_env import HopperEnv
from rllab.envs.mujoco.humanoid_env import HumanoidEnv
from rllab.envs.mujoco.simple_humanoid_env import SimpleHumanoidEnv
from rllab.envs.mujoco.maze.point_maze_env import PointMazeEnv
from rllab.envs.mujoco.maze.swimmer_maze_env import SwimmerMazeEnv
from rllab.envs.mujoco.maze.ant_maze_env import AntMazeEnv
from rllab.envs.mujoco.inverted_double_pendulum_env import InvertedDoublePendulumEnv
from rllab.misc import ext
from rllab.envs.normalized_env import normalize

from common import *


def create_env(env_str, framework='gym', args=None, eval_flag=False, norm=True,
               rank=0):
    if framework == 'gym':
        env = gym.make(env_str)
        if norm:
            env = NormalizedEnv(env)
    elif framework == 'rllab':
        if not hasattr(args, 'file_path'):
            args.file_path = None
        if env_str.endswith('MazeEnv'):
            if not hasattr(args, 'coef_inner_rew'):
                args.coef_inner_rew = 0.
            if not hasattr(args, 'maze_structure'):
                args.maze_structure = None
            if not hasattr(args, 'random_start'):
                args.random_start = False
            if not hasattr(args, 'difficulty'):
                args.difficulty = -1
            difficulty = args.difficulty
            if args.difficulty > 1 and not eval_flag:
                if args.difficulty <= 5:
                    difficulty = np.random.choice(range(
                        args.difficulty - 1, args.difficulty + 1))
                elif args.difficulty == -1:
                    difficulty = np.random.choice([1, 2, 3, 4, 5, -1])
            env = eval(env_str)(maze_id=args.maze_id, length=args.maze_length,
                                coef_inner_rew=args.coef_inner_rew,
                                structure=args.maze_structure,
                                file_path=args.file_path,
                                random_start=args.random_start,
                                difficulty=difficulty)
            env.horizon = args.max_episode_length
            vlog(args.maze_structure, args.v)
        else:
            env = eval(env_str)(file_path=args.file_path)
        if norm:
            env = normalize(env)
    else:
        raise("framework not supported")
    env.reset()
    set_seed(args.seed + rank, env, framework)
    return env


def wrapper(env):
    def _wrap():
        return env
    return _wrap


def get_obs(env, framework):
    if framework == 'gym':
        state = env.unwrapped._get_obs()
    elif framework == 'rllab':
        state = env.get_current_obs()
    else:
        raise("framework not supported")
    return state


def set_seed(seed, env, framework):
    if framework == 'gym':
        env.unwrapped.seed(seed)
    elif framework == 'rllab':
        ext.set_seed(seed)
    else:
        raise("framework not supported")
    return env


def reset_env(env, args):
    """Reset env. Can differ based on env. e.g. in maze maybe we want to randomly
    deposit the agent in different locations?"""
    env.reset()
    return get_obs(env, args.framework)


class NormalizedEnv(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizedEnv, self).__init__(env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def _observation(self, observation):
        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha + \
            observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + \
            observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        return (observation - unbiased_mean) / (unbiased_std + 1e-8)
