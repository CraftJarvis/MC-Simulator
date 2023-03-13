import gym
import gym.spaces as gym_spaces
import numpy as np
import os
import random
import logging
from copy import deepcopy

# These importing could take long
import minedojo
import minedojo.sim.spaces as mdj_spaces
from minedojo.sim import InventoryItem
from minedojo.sim.mc_meta.mc import ALL_ITEMS

MineDojoEnvList = [
    "Plains", 
    "Forest",
]


class MineDojoEnv(gym.Env):
    def __init__(
        self,
        name=None,
        img_size=(640, 480),
        rgb_only=False,
        ):
        if name not in MineDojoEnvList:
            print(f'{name} not in env list. Aborted.')
            assert False

        self.rgb_only = rgb_only
        if name == "Plains":
            self._env = minedojo.make(
                task_id = "harvest",
                image_size = list(img_size)[::-1],
                initial_mob_spawn_range_low = (-30, 1, -30),
                initial_mob_spawn_range_high = (30, 3, 30),
                initial_mobs = ["sheep", "cow", "pig", "chicken"] * 4,
                target_names = ["sheep", "cow", "pig", "chicken", "log"],
                # snow_golem
                target_quantities = 1,
                reward_weights = 1,
                initial_inventory = [],
                fast_reset_random_teleport_range = 100,
                # start_at_night = True,
                no_daylight_cycle = True,
                specified_biome = "plains",
                # generate_world_type = "flat",
                max_nsteps = 1000,
                need_all_success = False,
                voxel_size = dict(xmin=-1,ymin=0,zmin=1,xmax=1,ymax=1,zmax=2),
                use_voxel = True,
                custom_commands = ["/give @p minecraft:diamond_axe 1 0"],
                force_slow_reset_interval = 2,
            )
        elif name == "Forest": 
            self._env = minedojo.make(
                task_id = "harvest",
                image_size = list(img_size)[::-1],
                initial_mob_spawn_range_low = (-20, 1, -20),
                initial_mob_spawn_range_high = (20, 3, 20),
                initial_mobs = ["sheep", ] * 4,
                target_names = ["wool", "Oak Leaves", "Birch Leaves", "Spruce Leaves", "Grass", 
                                "sheep", "cow", "pig", "dirt", "stone", "sand", "Oak Wood", "Oak Sapling", "Birch Wood", "Birch Sapling", "Spruce Wood", "Spruce Sapling",
                                "Dandelion", "Poppy", "Blue Orchid", "Allium", "Azure Bluet", "Red Tulip", "Orange Tulip", "White Tulip", "Pink Tulip", "Oxeye Daisy", "Brown Mushroom",
                                "Pumpkin", "Sunflower", "Lilac", "Double Tallgrass", "Large Fern", "Rose Bush", "Peony", "Wheat seeds", "Sugar Canes",],
                # snow_golem
                target_quantities = 1,
                reward_weights = 1,
                initial_inventory = [],
                fast_reset_random_teleport_range = 100,
                # start_at_night = True,
                no_daylight_cycle = True,
                specified_biome = "flower_forest",
                # generate_world_type = "flat",
                max_nsteps = 600,
                need_all_success = False,
                voxel_size = dict(xmin=-1,ymin=0,zmin=1,xmax=1,ymax=1,zmax=2),
                use_voxel = True,
                custom_commands = ["/give @p minecraft:diamond_axe 1 0"],
                force_slow_reset_interval = 50,
            )
        else:
            assert False

        self.action_space = mdj_spaces.MultiDiscrete([3, 3, 4, 11, 11, 8, 1, 1], noop_vec=[0, 0, 0, 5, 5, 0, 0, 0]) # minedojo with VPT camera view

        # CHW -> HWC
        # FIXME
        c, h, w = self._env.observation_space['rgb'].shape
        high = self._env.observation_space['rgb'].high.max()
        low = self._env.observation_space['rgb'].low.min()
        self._env.observation_space['rgb'] = mdj_spaces.Box(shape=(h, w, c), high=high, low=low)

        self.text_obs_list = []
        self.all_text = ALL_ITEMS
        if self.rgb_only:
            self.observation_space = self._env.observation_space['rgb']
        else:
            obs_dict = {
                'rgb': self._env.observation_space['rgb'],
                'voxels': self._env.observation_space['voxels']['block_meta'],
                'gps': self._env.observation_space['location_stats']['pos'],
                'compass': mdj_spaces.Box(shape=(2, ), high=180.0 * 2, low=-180.0 * 2),
                'biome_id': self._env.observation_space['location_stats']['biome_id'],
            }
            self.observation_space = gym_spaces.Dict(obs_dict)

    def _discrete_to_multi(self, a):
        assert False # we're using modified action
        new_a = np.zeros(8).astype(np.int64)
        if 0 <= a < 3:
            new_a[0] = a
        elif 3 <= a < 6:
            new_a[1] = a-3
        elif 6 <= a < 10:
            new_a[2] = a-6
        elif 10 <= a < 35:
            new_a[3] = a-10
        elif 35 <= a < 60:
            new_a[4] = a-35
        elif 60 <= a < 68:
            new_a[5] = a-60
        elif 68 <= a < 312:
            new_a[6] = a-68
        elif 312 <= a < 348:
            new_a[7] = a-312
        else:
            assert False
        return new_a

    def close(self):
        self._env.close()
        super().close()

    def replace_text(self, obs):
        for i, l in self.text_obs_list:
            i = i.split('#')
            if len(i) == 1:
                text = obs[i[0]]
                if l == 1:
                    obs[i[0]] = np.array(self.all_text.index(text))
                else:
                    obs[i[0]] = np.array([self.all_text.index(t) for t in text])
            elif len(i) == 2:
                text = obs[i[0]][i[1]]
                if l == 1:
                    obs[i[0]][i[1]] = np.array(self.all_text.index(text))
                else:
                    obs[i[0]][i[1]] = np.array([self.all_text.index(t) for t in text])
            else:
                assert False
        return obs

    def build_obs(self, inp_obs):
        obs = {
            'rgb': inp_obs['rgb'], 
            'voxels': inp_obs['voxels']['block_meta'], 
            'gps': inp_obs['location_stats']['pos'], 
            'compass': np.concatenate([inp_obs['location_stats']['yaw'], inp_obs['location_stats']['pitch']]), 
            'biome_id': inp_obs['location_stats']['biome_id'], 
        }
        return obs
    
    def reset(self, task_name = None):
        if task_name is None:
            ret = self._env.reset()
        else:
            ret = self._env.reset(task_name = task_name)
        ret = self.replace_text(ret)
        ret['rgb'] = np.ascontiguousarray(ret['rgb'].transpose(1, 2, 0))
        if self.rgb_only:
            ret = ret['rgb']
        else:
            ret = self.build_obs(inp_obs=ret)
        return ret

    def step(self, action):
        # if self.fix_action:
        #     action = self._discrete_to_multi(action)
        obs, reward, done, info = self._env.step(action)
        obs = self.replace_text(obs)
        obs['rgb'] = np.ascontiguousarray(obs['rgb'].transpose(1, 2, 0))
        if self.rgb_only:
            obs = obs['rgb']
        else:
            obs = self.build_obs(inp_obs=obs)
        return obs, reward, done, info

    def seed(self, seed=None):
        return self._env.seed(seed)
