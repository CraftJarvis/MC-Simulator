import statistics
from typing import Optional, Union, List, Tuple, Dict

from .base import ExtraSpawnMetaTaskBase
from ...sim.inventory import InventoryItem
from .extra_spawn import Target2SpawnItem, SpawnItem2Condition
from .utils import (
    always_satisfy_condition,
    # simple_inventory_based_check,
    # simple_stat_kill_entity_based_check,
    # complex_inventory_based_check,
    # simple_inventory_based_reward,
    # simple_stat_kill_entity_based_reward,
    # complex_inventory_based_reward,
    inventory_based_check,
    inventory_based_reward,
    nearby_has_item_reward,
    pos_change_reward,
    # simple_inventory_based_accomplishments,
    # simple_kill_entity_based_accomplishments,
    # complex_inventory_based_accomplishments,
    inventory_based_accomplishments,
    max_timesteps_check
)
from .func_utils import get_reward_fns, get_success_criteria


class HarvestMeta(ExtraSpawnMetaTaskBase):
    """
    Class for harvest tasks.
    Args:
        allow_time_passage: Time flows if ``True``.
                Default: ``True``.

        break_speed_multiplier: Controls the speed of breaking blocks. A value larger than 1.0 accelerates the breaking.
                Default: ``1.0``.

        event_level_control: If ``True``, the agent is able to perform high-level controls including place and equip.
                If ``False``, then is keyboard-mouse level control.
                Default: ``True``.

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

        image_size: The size of image observations.

        initial_mobs: The types of mobs that are spawned initially.
                Default: ``None``.

        initial_mob_spawn_range_high: The upper bound on each horizontal axis from the center of the area to spawn initially.
                Default: ``None``.

        initial_mob_spawn_range_low: The lower bound on each horizontal axis from the center of the area to spawn initially.
                Default: ``None``.

        initial_weather: If not ``None``, specifies the initial weather.
                Can be one of ``"clear"``, ``"normal"``, ``"rain"``, ``"thunder"``.
                Default: ``None``.

        lidar_rays: Defines the directions and maximum distances of the lidar rays if ``use_lidar`` is ``True``.
                If supplied, should be a list of tuple(pitch, yaw, distance).
                Pitch and yaw are in radians and relative to agent looking vector.
                Default: ``None``.

        reward_weights: The reward weight for each target in the task.
                Default: ``1.0``.

        seed: The seed for an instance's internal generator.
                Default: ``None``.

        sim_name: Name of a simulation instance.
                Default: "HarvestMeta".

        spawn_range_high: The upper bound on each horizontal axis from the center of the area to spawn
                Default: ``None``.

        spawn_range_low: The lower bound on each horizontal axis from the center of the area to spawn
                Default: ``None``.

        spawn_rate: The probability of spawning in each step.
                Default: ``None``.

        specified_biome: The specified biome of the task.
                Default: ``None``.

        start_at_night: If ``True``, the task starts at night.
                Default: ``True``.

        start_food: If not ``None``, specifies initial food condition of the agent.
                Default: ``None``.

        start_health: If not ``None``, specifies initial health condition of the agent.
                Default: ``None``.

        start_position: If not ``None``, specifies the agent's initial location and orientation in the minecraft world.
                Default: ``None``.

        target_names: Names of target items to be harvested.

        target_quantities: The quantity of each target to be harvested.

        use_lidar: If ``True``, includes lidar in observations.
                Default: ``False``.

        use_voxel: If ``True``, includes voxel in observations.
                Default: ``False``.

        voxel_size: Defines the voxel's range in each axis if ``use_voxel`` is ``True``.
                If supplied, should be a dict with keys ``xmin``, ``xmax``, ``ymin``, ``ymax``, ``zmin``, ``zmax``.
                Each value specifies the voxel size relative to the agent.
                Default: ``None``.

        world_seed: The seed for generating a minecraft world if ``generate_world_type`` is ``"default"``.
                See `here <https://minecraft.wiki/w/Seed_(level_generation)>`_ for more details.
                Default: ``None``.
    """

    _prompt_template = "Harvest {targets}."

    def __init__(
        self,
        *,
        # ------ harvest targets, quantities, and reward weights
        target_names: Optional[Union[str, List[str]]] = None,
        target_quantities: Optional[Union[int, List[int], Dict[str, int]]] = None,
        reward_weights: Optional[Union[int, float, Dict[str, Union[int, float]]]] = 1.0,
        # ------ nearby targets 
        nearby_target_names: Optional[Union[str, List[str]]] = None,
        nearby_reward_weights: Optional[Union[int, float, Dict[str, Union[int, float]]]] = None,
        nearby_target_quantities: Optional[Union[int, List[int], Dict[str, int]]] = None,
        # ------ pos targets
        pos_targets: Optional[Union[str, List[str]]] = None,
        pos_dirs: Optional[Union[str, List[str]]] = None,
        pos_target_weights: Optional[Union[int, float, List[Union[int, float]]]] = None,
        # ------ initial & extra spawn control
        initial_mobs: Optional[Union[str, List[str]]] = None,
        initial_mob_spawn_range_low: Optional[Union[Tuple[int, int, int], List[Tuple[int, int, int]]]] = None,
        initial_mob_spawn_range_high: Optional[Union[Tuple[int, int, int], List[Tuple[int, int, int]]]] = None,
        random_shuffle_spawn_range: bool = False,
        spawn_rate: Optional[Union[float, List[float], Dict[str, float]]] = None,
        spawn_range_low: Optional[Tuple[int, int, int]] = None,
        spawn_range_high: Optional[Tuple[int, int, int]] = None,
        # ------ initial conditions ------
        initial_inventory: Optional[List[InventoryItem]] = None,
        start_position: Optional[Dict[str, Union[float, int]]] = None,
        start_health: Optional[float] = None,
        start_food: Optional[int] = None,
        start_at_night: bool = False,
        initial_weather: Optional[str] = None,
        # ------ global conditions ------
        allow_time_passage: bool = True,
        specified_biome: Optional[Union[int, str]] = None,
        break_speed_multiplier: float = 1.0,
        # ------ sim seed & world seed
        seed: Optional[int] = None,
        world_seed: Optional[str] = None,
        # ------ reset mode ------
        fast_reset: bool = True,
        # ------ obs ------
        image_size: Union[int, Tuple[int, int]],
        use_voxel: bool = False,
        voxel_size: Optional[Dict[str, int]] = None,
        use_lidar: bool = False,
        lidar_rays: Optional[List[Tuple[float, float, float]]] = None,
        # ------ event-level action or keyboard-mouse level action ------
        event_level_control: bool = True,
        # ------ misc ------
        sim_name: str = "HarvestMeta",
        fast_reset_random_teleport_range = 1000,
        force_slow_reset_interval = 0,
        no_daylight_cycle = True,
        generate_world_type = None,
        max_nsteps: int = 0,
        need_all_success: bool = True,
        custom_commands: Optional[List] = None,
        # ------ multi-task ------
        multi_task_specs: Optional[Dict] = None
    ):
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
            self._target_quantities = target_quantities
        else:
            self._target_quantities = None

        if isinstance(nearby_target_names, str):
            nearby_target_names = [nearby_target_names]
        if nearby_target_names is not None:
            if isinstance(nearby_reward_weights, int) or isinstance(nearby_reward_weights, float):
                nearby_reward_weights = {k: nearby_reward_weights for k in nearby_target_names}
            elif isinstance(nearby_reward_weights, dict):
                assert set(nearby_target_names) == set(nearby_reward_weights.keys())
            else:
                assert nearby_reward_weights is None
                nearby_reward_weights = {k: 1.0 for k in nearby_target_names}
        if isinstance(nearby_target_quantities, int):
            nearby_target_quantities = {k: nearby_target_quantities for k in nearby_target_names}
        elif isinstance(nearby_target_quantities, list):
            nearby_target_quantities = {k: nearby_target_quantities[i] for i, k in enumerate(nearby_target_names)}
        self._nearby_target_quantities = nearby_target_quantities if nearby_target_names is not None else None

        if isinstance(pos_targets, str):
            pos_targets = [pos_targets]
        if isinstance(pos_dirs, str):
            pos_dirs = [pos_dirs]
        if isinstance(pos_target_weights, int) or isinstance(pos_target_weights, float):
            pos_and_weights = {pos: (pos_dirs[i], 0.0, pos_target_weights) for i, pos in enumerate(pos_targets)}
        elif isinstance(pos_target_weights, list):
            pos_and_weights = {pos: (pos_dirs[i], 0.0, pos_target_weights[i]) for i, pos in enumerate(pos_targets)}
        self._pos_and_weights = pos_and_weights if pos_targets is not None else None

        spawn_condition = None
        if spawn_rate is not None:
            if isinstance(spawn_rate, float) or isinstance(spawn_rate, int):
                spawn_rate = {Target2SpawnItem[k]: spawn_rate for k in target_names}
            elif isinstance(spawn_rate, list):
                assert len(spawn_rate) == len(target_names)
                spawn_rate = {
                    Target2SpawnItem[k]: spawn_rate[i]
                    for i, k in enumerate(target_names)
                }
            elif isinstance(spawn_rate, dict):
                # don't do any checks here, so users can specify arbitrary spawn rates
                pass
            spawn_condition = {
                k: SpawnItem2Condition.get(k, always_satisfy_condition)
                for k in spawn_rate.keys()
            }

        reward_fns = get_reward_fns(target_names, reward_weights)

        success_criteria = get_success_criteria(target_names, target_quantities, max_nsteps)
            
        if nearby_target_names is not None:
            reward_fns += [nearby_has_item_reward(items_and_weights = nearby_reward_weights)]
        if pos_targets is not None:
            reward_fns += [pos_change_reward(pos_and_weights = self._pos_and_weights)]

        accomplishment_fns = []
        if target_names is not None:
            names = [name for name in reward_weights.keys()]
            accomplishment_fns += [
                # simple_inventory_based_accomplishments(names = names),
                # simple_kill_entity_based_accomplishments(names = names),
                # complex_inventory_based_accomplishments(names = names),
                inventory_based_accomplishments(names = names),
            ]

        start_time = 18000 if start_at_night else None

        if generate_world_type is None:
            generate_world_type = (
                "default" if specified_biome is None else "specified_biome"
            )

        super().__init__(
            initial_mobs=initial_mobs,
            initial_mob_spawn_range_low=initial_mob_spawn_range_low,
            initial_mob_spawn_range_high=initial_mob_spawn_range_high,
            random_shuffle_spawn_range=random_shuffle_spawn_range,
            extra_spawn_rate=spawn_rate,
            extra_spawn_condition=spawn_condition,
            extra_spawn_range_low=spawn_range_low,
            extra_spawn_range_high=spawn_range_high,
            fast_reset=fast_reset,
            success_criteria=success_criteria,
            reward_fns=reward_fns,
            accomplishment_fns=accomplishment_fns,
            seed=seed,
            sim_name=sim_name,
            image_size=image_size,
            use_voxel=use_voxel,
            voxel_size=voxel_size,
            use_lidar=use_lidar,
            lidar_rays=lidar_rays,
            event_level_control=event_level_control,
            initial_inventory=initial_inventory,
            break_speed_multiplier=break_speed_multiplier,
            world_seed=world_seed,
            start_position=start_position,
            initial_weather=initial_weather,
            start_time=start_time,
            allow_time_passage=allow_time_passage,
            start_health=start_health,
            start_food=start_food,
            generate_world_type=generate_world_type,
            specified_biome=specified_biome,
            fast_reset_random_teleport_range=fast_reset_random_teleport_range,
            force_slow_reset_interval=force_slow_reset_interval,
            no_daylight_cycle=no_daylight_cycle,
            need_all_success=need_all_success,
            custom_commands=custom_commands,
            multi_task_specs=multi_task_specs
        )

    def reset(self, **kwargs):
        obs = super(HarvestMeta, self).reset(**kwargs)
        info = self.env.prev_info

        if self._pos_and_weights is not None:
            for pos, (dir, _, weight) in self._pos_and_weights.items():
                self._pos_and_weights[pos] = (dir, info[pos], weight)

        return obs

    @property
    def task_prompt(self) -> str:
        filling = ", ".join(
            [
                f"{v} {str(k).replace('_', ' ')}"
                for k, v in self._target_quantities.items()
            ]
        )
        return super().get_prompt(targets=filling)
