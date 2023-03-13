#!/usr/bin/env python
# -*- coding:UTF-8 -*-


import minedojo
import time

import multiprocessing as mp
import time
import os
import argparse

import gym

from yaml import parse

from minedojo.minedojo_wrapper import MineDojoEnv
from minedojo.sim.wrappers import SafeEnvWrapper
from minedojo.sim import InventoryItem

def f(pid, return_dict):
    env = MineDojoEnv(name='Forest')
    env = SafeEnvWrapper(env)
    done = False
    obs = env.reset()
    count = 0
    s = time.time()
    #while True:
    for i in range(10000):
        if done:
            break
        count += 1
        # action = env.action_space.sample()
        action = env.action_space.no_op()
        obs, reward, done, info = env.step(action)
        res = count / (time.time() - s)
        if i % 100 == 0: 
            print(pid, res)

    return_dict[pid] = res
    env.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type = int, default = 1)

    args = parser.parse_args()

    SafeEnvWrapper.global_init()
    
    mp.set_start_method("forkserver")
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []
    for i in range(args.p):
        processes.append(mp.Process(target = f, args = (i, return_dict)))

    s = time.time()
    for p in processes:
        p.start()

    for p in processes:
        p.join()
    e = time.time()
    

    print(return_dict.values())
    print(f"mean fps: {sum(return_dict.values()) / args.p}")


