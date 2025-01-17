import copy
from collections import namedtuple
from itertools import count
import math
import random
import numpy as np
import time
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from collections import deque
import cv2

Transition = namedtuple('Transion',
                        ('state', 'action', 'next_state', 'reward'))
from environment import *
from memory import *
from model import *

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END)* \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state.to('cuda')).max(1)[1].view(1,1)
    else:
        return torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long)


def update_params():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    actions = tuple((map(lambda a: torch.tensor([[a]], device='cuda'), batch.action)))
    rewards = tuple((map(lambda r: torch.tensor([r], device='cuda'), batch.reward)))

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device, dtype=torch.bool)

    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None]).to('cuda')

    state_batch = torch.cat(batch.state).to('cuda')
    action_batch = torch.cat(actions)
    reward_batch = torch.cat(rewards)

    q_current = policy_net(state_batch).gather(1, action_batch)
    q_next = torch.zeros(BATCH_SIZE, device=device)
    q_next[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    q_target = (q_next * GAMMA) + reward_batch
    loss = F.smooth_l1_loss(q_current, q_target.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def get_state(obs):
    state = np.array(obs)
    state = state.transpose((2, 0, 1))
    state = torch.from_numpy(state)
    return state.unsqueeze(0)

def train(env, n_episodes):
    for episode in range(n_episodes):
        obs = env.reset()
        state = get_state(obs)
        total_reward = 0.0
        for t in count():
            action = select_action(state)
            obs, reward, done, info = env.step(action)
            total_reward += reward

            if not done:
                next_state = get_state(obs)
            else:
                next_state = None

            reward = torch.tensor([reward], device=device)
            memory.push(state, action.to('cpu'), next_state, reward.to('cpu'))
            state = next_state

            if steps_done > INITIAL_MEMORY:
                update_params()

                if steps_done % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            if done:
                break
        if episode % 20 == 0:
                print(' Total steps: {} \t Episode: {}/{} \t Total reward: {}'.format(steps_done, episode, t, total_reward))
    env.close()
    return

def test(env, n_episodes, policy, render=True):
    env = gym.wrappers.Monitor(env, './videos/' + 'playVids')
    for episode in range(n_episodes):
        obs = env.reset()
        state = get_state(obs)
        total_reward = 0.0
        for t in count():
            action = policy(state.to('cuda')).max(1)[1].view(1,1)
            obs, reward, done, info = env.step(action)
            total_reward += reward

            if not done:
                next_state = get_state(obs)
            else:
                next_state = None
            state = next_state

            if done:
                print("Finished Episode {} with reward {}".format(episode, total_reward))
                break

    env.close()
    return

if __name__ == '__main__':
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # hyperparameters
    BATCH_SIZE = 32
    GAMMA = 0.99
    EPS_START = 1
    EPS_END = 0.02
    EPS_DECAY = 200000
    TARGET_UPDATE = 1000
    RENDER = False
    lr = 1e-4
    INITIAL_MEMORY = 10000
    MEMORY_SIZE = 8 * INITIAL_MEMORY

    # create environment
    env = gym.make("PongNoFrameskip-v4")
    env = make_env(env)
    ACTION_DIM = env.action_space.n

    # create networks
    policy_net = DQN(n_actions=ACTION_DIM).to(device)
    target_net = DQN(n_actions=ACTION_DIM).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    # setup optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    steps_done = 0

    # initialize replay memory
    memory = ReplayMemory(MEMORY_SIZE)

    # train model
    train(env, 400)
    torch.save(policy_net, "Model")
    policy_net = torch.load("Model")
    test(env, 1, policy_net)
