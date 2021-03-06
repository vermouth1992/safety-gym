#!/usr/bin/env python

import argparse

import gym
import numpy as np  # noqa
import safety_gym  # noqa
from safety_gym.wrappers import CMDPWrapper


def run_random(env_name, render):
    env = gym.make(env_name)
    env = CMDPWrapper(env)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)
    obs = env.reset()
    done = False
    ep_ret = 0
    ep_cost = 0
    ep_len = 0
    while True:
        if done:
            print('Episode Return: %.3f \t Episode Cost: %.3f\t Episode Length: %d' % (ep_ret, ep_cost, ep_len))
            ep_ret, ep_cost, ep_len = 0, 0, 0
            obs = env.reset()
        if render:
            env.render()
        assert env.observation_space.contains(obs)
        act = env.action_space.sample()
        assert env.action_space.contains(act)
        obs, reward, done, info = env.step(act)
        # print('reward', reward)
        ep_ret += reward
        ep_cost += info.get('cost', 0)
        ep_len += 1
        env.render()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='Safexp-PointGoal1-v0')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    run_random(args.env, args.render)
