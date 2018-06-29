# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import time
from collections import deque

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from envs import create_env
from model import *


def test(rank, args, shared_model, counter):
    torch.manual_seed(args.seed + rank)

    env = create_env(args.env_name)
    env.seed(args.seed + rank)

    model = ActorCritic(env.observation_space.shape[0], env.action_space)

    model.eval()

    state = env.reset()
    state = torch.from_numpy(state).float()
    reward_sum = 0
    done = True

    start_time = time.time()

    # a quick hack to prevent the agent from stucking
    actions = deque(maxlen=100)
    episode_length = 0
    while True:
        episode_length += 1
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())
            cx_d = Variable(torch.zeros(1, 256), volatile=True)
            hx_d = Variable(torch.zeros(1, 256), volatile=True)
            cx_p = Variable(torch.zeros(1, 256), volatile=True)
            hx_p = Variable(torch.zeros(1, 256), volatile=True)
        else:
            cx_d = Variable(cx_d.data, volatile=True)
            hx_d = Variable(hx_d.data, volatile=True)
            cx_p = Variable(cx_p.data, volatile=True)
            hx_p = Variable(hx_p.data, volatile=True)

        value, logit, (hx_d, cx_d), (hx_p, cx_p) = model((Variable(
            state.unsqueeze(0), volatile=True), (hx_d, cx_d), (hx_p, cx_p)))
        if args.discrete:
            prob = F.softmax(logit)
            action = prob.max(1, keepdim=True)[1].data.numpy()
        else:
            mu, sigma_sq = logit
            sigma_sq = F.softplus(sigma_sq)
            eps = torch.randn(mu.size())
            action = (mu + sigma_sq.sqrt()*Variable(eps)).data
        state, reward, done, _ = env.step(action[0, 0])
        done = done or episode_length >= args.max_episode_length
        reward_sum += reward

        # a quick hack to prevent the agent from stucking
        actions.append(action[0, 0])
        if actions.count(actions[0]) == actions.maxlen:
            done = True

        if done:
            print("Time {}, num steps {}, FPS {:.0f}, episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                counter.value, counter.value / (time.time() - start_time),
                reward_sum, episode_length))
            reward_sum = 0
            episode_length = 0
            actions.clear()
            state = env.reset()
            time.sleep(60)

        state = torch.from_numpy(state).float()
