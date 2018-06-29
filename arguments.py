# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Train Modules')
    # Learning parameters
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='parameter for GAE (default: 0.95)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--max-grad-norm', type=float, default=50,
                        help='value loss coefficient (default: 50)')
    parser.add_argument('--no-shared', default=False,
                        help='use an optimizer without shared momentum.')
    parser.add_argument('--dim', type=int, default=32,
                        help='number of dimensions of representation space')
    parser.add_argument('--use-conv', action='store_true', help='Use conv layers')
    parser.add_argument('--discrete', action='store_true', help='discrete action space')
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    # TODO:// finish implementation for discrete action spaces.

    # Environment settings
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--num-processes', type=int, default=40,
                        help='how many training processes to use (default: 40)')
    parser.add_argument('--num-steps', type=int, default=200,
                        help='number of forward steps in A3C (default: 20)')
    parser.add_argument('--framework', default='gym',
                        help='framework of env (default: gym)')
    parser.add_argument('--env-name', default='InvertedPendulum-v1',
                        help='environment to train on (default: InvertedPendulum-v1)')
    parser.add_argument('--maze-id', type=int, default=0)
    parser.add_argument('--maze-length', type=int, default=1)

    # Dynamics Module settings
    parser.add_argument('--rollout', type=int, default=20, help="rollout for goal")
    parser.add_argument('--train-set', type=str, default=None)
    parser.add_argument('--train-batch', type=int, default=2500)
    parser.add_argument('--test-set', type=str)
    parser.add_argument('--test-batch', type=int, default=2500)
    parser.add_argument('--train-size', type=int, default=100000)
    parser.add_argument('--dec-loss-coef', type=float, default=0.1,
                        help='decoder loss coefficient (default: 0.1)')
    parser.add_argument('--forward-loss-coef', type=float, default=10,
                        help='forward loss coefficient (default: 10)')
    parser.add_argument('--inv-loss-coef', type=float, default=100,
                        help='inverse loss coefficient (default: 10)')
    parser.add_argument('--num-epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=20)
    parser.add_argument('--out', type=str, default='/checkpoint/amyzhang/ddr/models')
    parser.add_argument('--dec-mask', type=float, default = None,
                     help="to use masking while calculating the decoder reconstruction loss ")

    # Rewards Module settings
    parser.add_argument('--coef-inner-rew', type=float, default=1.)
    parser.add_argument('--checkpoint-interval', type=int, default=1000)
    parser.add_argument('--num-episodes', type=int, default=1000000,
                        help='max number of episodes to train')
    parser.add_argument('--max-episode-length', type=int, default=500,
                        help='maximum length of an episode (default: 500)')
    parser.add_argument('--curriculum', type=int, default=0,
        help='number of iterations in curriculum. (default: 0, no curriculum)')
    parser.add_argument('--single-env', action='store_true')
    parser.add_argument('--entropy-coef', type=float, default=0.,
                        help='entropy term coefficient (default: 0.), use 0.0001 for mujoco')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--rew-loss-coef', type=float, default=0,
                        help='reward loss coefficient (default: 0)')
    parser.add_argument('--lstm-dim', type=int, default=128,
                        help='number of dimensions of lstm hidden state')
    parser.add_argument('--difficulty', type=int, default=-1, help='difficulty of maze')
    parser.add_argument('--clip-reward', action='store_true')
    parser.add_argument('--finetune-enc', action='store_true',
                help="allow the ActorCritic to change the observation space representation")
    parser.add_argument('--gae', action='store_true')
    parser.add_argument('--algo', default='a3c',
                        help='algorithm to use: a3c')

    # General training settings
    parser.add_argument('--checkpoint', type=int, default=10000)
    parser.add_argument('--log-interval', type=int, default=100,
                        help='interval between training status logs (default: 100)')
    parser.add_argument('-v', action='store_true', help='verbose logging')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--log-dir', type=str, default='/checkpoint/amyzhang/ddr/logs',
                        help='The logging directory to record the logs and tensorboard summaries')
    parser.add_argument('--reset-dir', action='store_true',
                        help="give this argument to delete the existing logs for the current set of parameters")

    # transfer
    parser.add_argument('--file-path', type=str, default=None,
                        help='path to XML file for mujoco')
    parser.add_argument('--neg-reward', action='store_true',
                        help='set reward negative for transfer')
    parser.add_argument('--random-start', action='store_true')

    # What to run
    parser.add_argument('--train-dynamics', action='store_true')
    parser.add_argument('--train-reward', action='store_true')
    parser.add_argument('--train-online', action='store_true',
                        help='train both modules online')
    parser.add_argument('--dynamics-module', type=str, default=None,
                        help='Encoder from dynamics module')
    parser.add_argument('--from-checkpoint', type=str, default=None,
                        help='Start from stored model')
    parser.add_argument('--baseline', action='store_true',
                        help='Running A3C baseline.')
    parser.add_argument('--planning', action='store_true',
                        help='train with planning (reward and online only)')
    parser.add_argument('--transfer', action='store_true',
                        help='Keep encoder and decoder static')
    parser.add_argument('--eval-every', type=float, default=10)
    parser.add_argument('--enc-dims', type=int, nargs='+', default=[256, 128])
    parser.add_argument('--dec-dims', type=int, nargs='+', default=[128, 256])
    parser.add_argument('--num-runs', type=int, default=5,
                        help='number of models to train in parallel')
    parser.add_argument('--mcts', action='store_true', help='Monte Carlo Tree Search')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('-b', type=int, default=4, help='branching factor')
    parser.add_argument('-d', type=int, default=3, help='planning depth')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--local', action='store_true')

    args = parser.parse_args()
    return args
