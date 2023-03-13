from __future__ import annotations

from copy import deepcopy
from typing import List, Dict, Any, Union, Tuple, Optional

import gym
import numpy as np
import sys

from ...sim import MineDojoSim
from ...sim.wrappers import FastResetWrapper
from .utils import (
    check_success_base,
    reward_fn_base,
    extra_spawn_condition_base,
    always_satisfy_condition,
    accomplishment_fn_base
)


__all__ = ["MetaTaskBase", "ExtraSpawnMetaTaskBase"]


class MetaTaskBase(gym.Wrapper):
    """
    Base class for all meta tasks.

    Args:
        fast_reset: If ``True``, reset using MC native command `/kill`, instead of waiting for re-generating new worlds.
            Default: ``True``.

            .. warning::
                Side effects:

                1. Changes to the world will not be reset. E.g., if the agent chops lots of trees then calling
                fast reset will not restore those trees.

                2. If you specify agent starting health and food, these specs will only be respected at the first reset
                (i.e., generating a new world) but will not be respected in the following resets (i.e., reset using MC cmds).
                So be careful to use this wrapper if your usages require specific initial health/food.

                3. Statistics/achievements will not be reset. This wrapper will maintain a property ``info_prev_reset``.
                If your tasks use stat/achievements to evaluation, please retrieve this property and compute differences.

        reward_fns: The reward functions of the task.
        success_criteria: The success criteria of the task.
    """

    _prompt_template: str
    _use_specified_prompt: bool = False
    _specified_prompt: str | None = None
    _guidance: str | None = None

    def __init__(
        self,
        *,
        fast_reset: bool = True,
        fast_reset_random_teleport_range: Optional[int] = None,
        no_daylight_cycle = True,
        success_criteria: List[check_success_base],
        reward_fns: List[reward_fn_base],
        accomplishment_fns: List[accomplishment_fn_base],
        need_all_success: bool,
        custom_commands: Optional[List] = None,
        force_slow_reset_interval: int = 0,
        multi_task_specs: Optional[Dict] = None,
        **kwargs,
    ):
        sim = MineDojoSim(**kwargs)
        self._fast_reset = fast_reset
        if fast_reset:
            sim = FastResetWrapper(
                sim, random_teleport_range=fast_reset_random_teleport_range,
                random_teleport_agent = (fast_reset_random_teleport_range > 0),
                no_daylight_cycle = no_daylight_cycle,
                custom_commands = custom_commands,
                force_slow_reset_interval = force_slow_reset_interval
            )
        # if multi_task_specs is not None:
        #     sim = MultiTaskWrapper(
        #         sim, multi_task_specs
        #     )
        super().__init__(env=sim)

        self._ini_info_dict = None
        self._pre_info_dict = None
        self._elapsed_timesteps = None
        self._is_successful: bool = False

        self._success_criteria = success_criteria
        self._reward_fns = reward_fns
        self._accomplishment_fns = accomplishment_fns

        self.need_all_success = need_all_success

    @property
    def is_successful(self):
        return self._is_successful

    def get_success_vector(self):
        return [
                check_success(
                    ini_info_dict=self._ini_info_dict,
                    cur_info_dict=self._pre_info_dict,
                    elapsed_timesteps=self._elapsed_timesteps,
                )
                for check_success in self._success_criteria
            ] # add for check multi-item whether has been done.

    def reset(self, updated_reward_fns = None, updated_success_criteria = None, **kwargs):
        """Resets the environment to an initial state and returns an initial observation.

        Return:
            Agent’s initial observation.
        """
        if updated_reward_fns is not None:
            self._reward_fns = updated_reward_fns
        if updated_success_criteria is not None:
            self._success_criteria = updated_success_criteria

        obs = self.env.reset(**kwargs)
        info = self.env.prev_info
        obs, info = self._after_sim_reset_hook(obs, info)
        self._ini_info_dict = (
            self.env.info_prev_reset or info if self._fast_reset else info
        )
        if "spawn_item_perm" in info:
            self._spawn_item_perm = info["spawn_item_perm"]
        self._pre_info_dict = deepcopy(info)
        self._elapsed_timesteps = 0
        self._is_successful = False
        # self._success_vector = False
        return obs

    def step(self, action):
        """Run one timestep of the environment’s dynamics. Accepts an action and returns next_obs, reward, done, info.

        Args:
            action: The action of the agent in current step.

        Return:
            A tuple (obs, reward, done, info)
            - ``dict`` - Agent’s next observation.
            - ``float`` - Amount of reward returned after executing previous action.
            - ``bool`` - Whether the episode has ended.
            - ``dict`` - Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        obs, _, _, info = self.env.step(action)
        self._elapsed_timesteps += 1
        # print(obs.keys())
        # print(obs)
        # info["voxels"] = obs["voxels"]
        if "voxels" in obs.keys():
            info["voxels"] = obs["voxels"]
        reward = self._compute_reward_hook(
            ini_info=self._ini_info_dict,
            pre_info=self._pre_info_dict,
            cur_info=info,
            elapsed_timesteps=self._elapsed_timesteps,
        )
        self._is_successful = self._determine_success_hook(
            ini_info=self._ini_info_dict,
            cur_info=info,
            elapsed_timesteps=self._elapsed_timesteps,
            need_all_success=self.need_all_success
        )
        # self._success_vector = [
        #             check_success(
        #                 ini_info_dict=self._ini_info_dict,
        #                 cur_info_dict=info,
        #                 elapsed_timesteps=self._elapsed_timesteps,
        #             )
        #             for check_success in self._success_criteria
        #         ] # add for check multi-item whether has been done.
        done = self.env.is_terminated or self._is_successful
        info["accomplishments"] = self._get_accomplishments_hook(
            ini_info=self._ini_info_dict,
            pre_info=self._pre_info_dict,
            cur_info=info,
            elapsed_timesteps=self._elapsed_timesteps,
        )
        self._pre_info_dict = deepcopy(info)
        return obs, reward, done, info

    def get_prompt(self, **kwargs) -> str:
        """Get the prompt of the task"""
        return (
            self._specified_prompt
            if self._use_specified_prompt
            else self._prompt_template.format(**kwargs)
        )

    def specify_prompt(self, prompt: str):
        """Allow users to customize the prompt"""
        self._specified_prompt = prompt
        self._use_specified_prompt = True

    @property
    def task_guidance(self):
        return self._guidance

    def specify_guidance(self, guidance: str):
        """Allow users to customize the guidance"""
        self._guidance = guidance

    def _after_sim_reset_hook(
        self, reset_obs: Dict[str, Any], reset_info: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Called immediately after `self.env.reset()` in `self.reset()`.
        Override to implement initial conditions, e.g., initial mob spawn, initial teleporting, etc.
        Takes input of obs and info from sim reset.
        Should return resulting obs and info.
        """
        return reset_obs, reset_info

    def _compute_reward_hook(
        self,
        ini_info: Dict[str, Any],
        pre_info: Dict[str, Any],
        cur_info: Dict[str, Any],
        elapsed_timesteps: int,
    ) -> Union[int, float]:
        return sum(
            [
                reward_fn(
                    ini_info_dict=ini_info,
                    pre_info_dict=pre_info,
                    cur_info_dict=cur_info,
                    elapsed_timesteps=elapsed_timesteps,
                )
                for reward_fn in self._reward_fns
            ]
        )

    def _determine_success_hook(
        self, ini_info: Dict[str, Any], cur_info: Dict[str, Any], elapsed_timesteps: int, need_all_success: bool = True
    ) -> bool:
        if need_all_success:
            return False if len(self._success_criteria) == 0 else all(
                [
                    check_success(
                        ini_info_dict=ini_info,
                        cur_info_dict=cur_info,
                        elapsed_timesteps=elapsed_timesteps,
                    )
                    for check_success in self._success_criteria
                ]
            )
        else:
            return False if len(self._success_criteria) == 0 else any(
                [
                    check_success(
                        ini_info_dict=ini_info,
                        cur_info_dict=cur_info,
                        elapsed_timesteps=elapsed_timesteps,
                    )
                    for check_success in self._success_criteria
                ]
            )
    
    def _get_accomplishments_hook(
        self, ini_info: Dict[str, Any], pre_info: Dict[str, Any], cur_info: Dict[str, Any], elapsed_timesteps: int
    ) -> List[str]:
        accomplishments = []
        for accomplishment_fn in self._accomplishment_fns:
            accomplishments += accomplishment_fn(
                ini_info_dict=ini_info,
                pre_info_dict=pre_info,
                cur_info_dict=cur_info,
                elapsed_timesteps=elapsed_timesteps
            )

        return accomplishments


class ExtraSpawnMetaTaskBase(MetaTaskBase):
    """
    Base class for all meta tasks that need extra spawn rate for resources or mobs.

    Args:
        extra_spawn_condition: The extra spawn rate applies only when the extra spawn conditions are satisfied.

        extra_spawn_range_high: The upper bound on each horizontal axis from the center of the area to extra spawn.

        extra_spawn_range_low: The lower bound on each horizontal axis from the center of the area to extra spawn.

        extra_spawn_rate: The probability of extra spawning in each step.
                Default: ``None``.

        fast_reset: If ``True``, fast reset will be used in the task. See ``fast_reset`` in ``MetaTaskBase``.
                Default: ``True``.

        initial_mobs: The types of mobs that are spawned initially.
                Default: ``None``.

        initial_mob_spawn_range_high: The upper bound on each horizontal axis from the center of the area to spawn initially.
                Default: ``None``.

        initial_mob_spawn_range_low: The lower bound on each horizontal axis from the center of the area to spawn initially.
                Default: ``None``.

        reward_fns: The reward functions of the task.

        success_criteria: The success criteria of the task.
    """

    # TODO (yunfan): complete these lists
    by_setblock = ["diamond_ore", "gold_ore", "iron_ore", "coal_ore"]
    by_summon = ["pig", "cow", "bat", "cat", "chicken", "horse", "sheep", "zombie"]

    def __init__(
        self,
        *,
        initial_mobs: Optional[Union[str, List[str]]] = None,
        initial_mob_spawn_range_low: Optional[Union[Tuple[int, int, int], List[Tuple[int, int, int]]]] = None,
        initial_mob_spawn_range_high: Optional[Union[Tuple[int, int, int], List[Tuple[int, int, int]]]] = None,
        random_shuffle_spawn_range: bool = False,
        extra_spawn_rate: Optional[Dict[str, float]] = None,
        extra_spawn_condition: Optional[
            Union[extra_spawn_condition_base, Dict[str, extra_spawn_condition_base]]
        ],
        extra_spawn_range_low: Optional[Tuple[int, int, int]],
        extra_spawn_range_high: Optional[Tuple[int, int, int]],
        fast_reset: bool = True,
        fast_reset_random_teleport_range: Optional[int] = None,
        success_criteria: List[check_success_base],
        reward_fns: List[reward_fn_base],
        accomplishment_fns: List[accomplishment_fn_base],
        need_all_success: Optional[bool] = None,
        custom_commands: Optional[List] = None,
        **kwargs,
    ):
        """
        base task class for meta tasks that require extra spawn, e.g., harvest, combat, and techtree
        """
        self._extra_spawn_rate = None
        if extra_spawn_rate is not None:
            assert all(
                [1.0 >= rate >= 0.0 for rate in extra_spawn_rate.values()]
            ), "extra spawn rate must <= 1.0 and >= 0.0"
            assert all(
                [
                    name in self.by_summon or name in self.by_setblock
                    for name in extra_spawn_rate.keys()
                ]
            ), f"{extra_spawn_rate.keys()} should belong to either {self.by_summony} or {self.by_setblock}"
            if extra_spawn_condition is None:
                extra_spawn_condition = {
                    k: always_satisfy_condition for k in extra_spawn_rate.keys()
                }
            else:
                if isinstance(extra_spawn_condition, dict):
                    assert set(extra_spawn_rate.keys()) == set(
                        extra_spawn_condition.keys()
                    ), (
                        f"extra_spawn_rate must match extra_spawn_condition, "
                        f"but got {extra_spawn_rate.keys()} and {extra_spawn_condition.keys()}"
                    )
                else:
                    extra_spawn_condition = {
                        k: extra_spawn_condition for k in extra_spawn_rate.keys()
                    }
            self._extra_spawn_rate = extra_spawn_rate
            self._extra_spawn_rates_and_conditions = {
                k: (extra_spawn_rate[k], extra_spawn_condition[k])
                for k in extra_spawn_rate.keys()
            }
            assert (
                extra_spawn_range_high is not None and extra_spawn_range_low is not None
            )
            assert len(extra_spawn_range_high) == 3 and len(extra_spawn_range_low) == 3
            low = np.repeat(
                np.array(extra_spawn_range_low)[np.newaxis, ...],
                len(extra_spawn_rate),
                axis=0,
            )
            high = np.repeat(
                np.array(extra_spawn_range_high)[np.newaxis, ...],
                len(extra_spawn_rate),
                axis=0,
            )
            self._extra_spawn_range_space = gym.spaces.Box(
                low=low, high=high, seed=kwargs["seed"]
            )
            self._rng = np.random.default_rng(seed=kwargs["seed"])

        super().__init__(
            fast_reset=fast_reset,
            fast_reset_random_teleport_range=fast_reset_random_teleport_range,
            success_criteria=success_criteria,
            reward_fns=reward_fns,
            accomplishment_fns=accomplishment_fns,
            need_all_success=True if need_all_success is None else need_all_success,
            custom_commands=custom_commands,
            **kwargs,
        )

        initial_mobs = initial_mobs or []
        if isinstance(initial_mobs, str):
            initial_mobs = [initial_mobs]
        self._initial_mobs = initial_mobs
        self._random_shuffle_spawn_range = random_shuffle_spawn_range
        if len(initial_mobs) > 0:
            if isinstance(initial_mob_spawn_range_low, Tuple):
                assert len(initial_mob_spawn_range_low) == 3
                assert len(initial_mob_spawn_range_high) == 3
                low = np.repeat(
                    np.array(initial_mob_spawn_range_low)[np.newaxis, ...],
                    len(initial_mobs),
                    axis=0,
                )
                high = np.repeat(
                    np.array(initial_mob_spawn_range_high)[np.newaxis, ...],
                    len(initial_mobs),
                    axis=0,
                )
                self._mob_spawn_range_spaces = [gym.spaces.Box(
                    low=low, high=high, seed=kwargs["seed"]
                )]
            elif isinstance(initial_mob_spawn_range_low, List):
                self._mob_spawn_range_spaces = []
                for range_low, range_high in zip(initial_mob_spawn_range_low, initial_mob_spawn_range_high):
                    low = np.repeat(
                        np.array(range_low)[np.newaxis, ...],
                        len(initial_mobs),
                        axis=0,
                    )
                    high = np.repeat(
                        np.array(range_high)[np.newaxis, ...],
                        len(initial_mobs),
                        axis=0,
                    )
                    self._mob_spawn_range_spaces.append(gym.spaces.Box(
                        low=low, high=high, seed=kwargs["seed"]
                    ))

    def step(self, action):
        if self._extra_spawn_rate is None:
            return super().step(action=action)
        else:
            rel_positions = self._extra_spawn_range_space.sample()
            for (name, (rate, condition)), pos in zip(
                self._extra_spawn_rates_and_conditions.items(), rel_positions
            ):
                if self._rng.random() <= rate and condition(
                    ini_info_dict=self._ini_info_dict,
                    pre_info_dict=self._pre_info_dict,
                    elapsed_timesteps=self._elapsed_timesteps,
                ):
                    if name in self.by_summon:
                        obs, _, _, info = self.env.spawn_mobs(name, pos)
                    elif name in self.by_setblock:
                        obs, _, _, info = self.env.set_block(name, pos)
            return super().step(action=action)

    def _after_sim_reset_hook(
        self, reset_obs: Dict[str, Any], reset_info: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        obs, info = reset_obs, reset_info
        m = None
        if len(self._initial_mobs) > 0:
            if not self._random_shuffle_spawn_range:
                mobs_rel_positions = self._mob_spawn_range_spaces[0].sample()
            else:
                mobs_rel_positions = np.zeros([len(self._initial_mobs), 3])
                m = np.random.permutation(len(self._initial_mobs))
                for i in range(len(self._initial_mobs)):
                    mobs_rel_positions[i,:] = self._mob_spawn_range_spaces[m[i]].sample()[0,:]

            obs, _, _, info = self.env.spawn_mobs(
                self._initial_mobs, mobs_rel_positions
            )
        if m is not None:
            info["spawn_item_perm"] = m
        return obs, info
