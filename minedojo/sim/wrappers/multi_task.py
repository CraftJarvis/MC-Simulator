import gym

from ..sim import MineDojoSim
from ...tasks.meta.func_utils import get_reward_fns, get_success_criteria
from typing import Dict

class MultiTaskWrapper(gym.Wrapper):
    def __init__(self,
                 env: MineDojoSim,
                 task_specs: Dict):
        super().__init__(env=env)

        self.task_specs = task_specs
        self.task_names = list(task_specs.keys())

        # Process `target_names`, `target_quantities`, `reward_weights`
        for task_name in self.task_names:
            task = self.task_specs[task_name]
            if "target_names" in task.keys():
                target_names = task["target_names"]
            else:
                target_names = None

            if "target_quantities" in task.keys():
                target_quantities = task["target_quantities"]
            else:
                target_quantities = None

            if "reward_weights" in task.keys():
                reward_weights = task["reward_weights"]
            else:
                reward_weights = None

            if "max_nsteps" in task.keys():
                max_nsteps = task["max_nsteps"]
            else:
                max_nsteps = 0

            if target_names is not None:
                if isinstance(target_names, str):
                    target_names = [target_names]
                if isinstance(target_quantities, int):
                    target_quantities = {k: target_quantities for k in target_names}
                elif isinstance(target_quantities, list):
                    assert len(target_names) == len(target_quantities)
                    target_quantities = {
                        k: target_quantities[i] for i, k in enumerate(target_names)
                    }
                elif isinstance(target_quantities, dict):
                    assert set(target_names) == set(target_quantities.keys())
                if isinstance(reward_weights, int) or isinstance(reward_weights, float):
                    reward_weights = {k: reward_weights for k in target_names}
                elif isinstance(reward_weights, dict):
                    assert set(target_names) == set(reward_weights.keys())
            else:
                target_quantities = None
                reward_weights = None

            task["target_names"] = target_names
            task["target_quantities"] = target_quantities
            task["reward_weights"] = reward_weights
            task["reward_fns"] = get_reward_fns(target_names, reward_weights)
            task["success_criteria"] = get_success_criteria(target_names, target_quantities, max_nsteps)

            self.task_specs[task_name] = task

        self.curr_task_name = None

    def reset(self, task_name = None):
        if task_name == None:
            task_name = self.task_names[0]
        else:
            assert task_name in self.task_names, "Unknown task {}".format(task_name)

        self.curr_task_name = task_name

        obs = self.env.reset(
            updated_reward_fns = self.task_specs[task_name]["reward_fns"],
            updated_success_criteria = self.task_specs[task_name]["success_criteria"]
        )

        # Execute custom commands
        if "custom_commands" in self.task_specs[task_name].keys():
            for cmd in self.task_specs[task_name]["custom_commands"]:
                obs, _, _, info = self.env.execute_cmd(cmd)

        return obs

    def step(self, action):
        obs, r, d, info = self.env.step(action)

        return obs, r, d, info

    def execute_cmd(self, *args, **kwargs):
        return self.env.execute_cmd(*args, **kwargs)

    def spawn_mobs(self, *args, **kwargs):
        return self.env.spawn_mobs(*args, **kwargs)

    def set_block(self, *args, **kwargs):
        return self.env.set_block(*args, **kwargs)

    def clear_inventory(self, *args, **kwargs):
        return self.env.clear_inventory(*args, **kwargs)

    def set_inventory(self, *args, **kwargs):
        return self.env.set_inventory(*args, **kwargs)

    def teleport_agent(self, *args, **kwargs):
        return self.env.teleport_agent(*args, **kwargs)

    def kill_agent(self, *args, **kwargs):
        return self.env.kill_agent(*args, **kwargs)

    def set_time(self, *args, **kwargs):
        return self.env.set_time(*args, **kwargs)

    def set_weather(self, *args, **kwargs):
        return self.env.set_weather(*args, **kwargs)

    def random_teleport(self, *args, **kwargs):
        return self.env.random_teleport(*args, **kwargs)

    @property
    def prev_obs(self):
        return self.env.prev_obs

    @property
    def prev_info(self):
        return self.env.prev_info

    @property
    def info_prev_reset(self):
        return self._info_prev_reset

    @property
    def prev_action(self):
        return self.env.prev_action

    @property
    def is_terminated(self):
        return self.env.is_terminated