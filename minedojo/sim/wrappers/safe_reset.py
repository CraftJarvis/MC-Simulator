import gym
import fcntl
import os
import time
import numpy as np


class SafeEnvWrapper(gym.Env):
    def __init__(self, env, temp_file_name = "/tmp/minerl-watcher.lock", max_slow_reset_procs = 4):
        self.env = env
        self.temp_file_name = temp_file_name
        self.max_slow_reset_procs = max_slow_reset_procs

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    @staticmethod
    def global_init(temp_file_name = "/tmp/minerl-watcher.lock"):
        f = open(temp_file_name, "a+")
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise ValueError("File already locked")
        f.truncate(0)
        f.write("0")
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()

    def _acquire_lock(self, file_name, mode = "a+"):
        f = open(file_name, mode)
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            f.close()
            return None
        return f
    
    def _release_lock(self, f):
        f.close()

    def reset(self, slow_reset = True):
        if slow_reset:
            # increase watcher file count
            while not os.path.exists(self.temp_file_name):
                time.sleep(0.1)
            allow_reset = False
            while not allow_reset:
                f = self._acquire_lock(self.temp_file_name)
                if f is not None:
                    f.seek(0)
                    num = int(f.read().strip("\n"))
                    f.seek(0, 2)
                    if num < self.max_slow_reset_procs:
                        f.truncate(0)
                        f.write(f"{num+1}")
                        allow_reset = True
                        time.sleep(0.1)
                    self._release_lock(f)

            obs = self.env.reset()

            # decrease watcher file count
            num_decreased = False
            while not num_decreased:
                f = self._acquire_lock(self.temp_file_name)
                if f is not None:
                    f.seek(0)
                    num = int(f.read().strip("\n"))
                    f.seek(0, 2)
                    f.truncate(0)
                    f.write(f"{num-1}")
                    num_decreased = True
                    self._release_lock(f)

        else:
            raise NotImplementedError()

        return obs

    def step(self, action):
        return self.env.step(action)