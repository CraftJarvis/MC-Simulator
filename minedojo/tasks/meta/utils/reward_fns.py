from functools import partial
from mypy_extensions import Arg
from typing import Union, Callable, Dict, Tuple
import json
import os 

with open(os.path.join(os.path.dirname(__file__), "names.json"),"r") as f:
    names_dict = json.load(f)


__all__ = [
    "reward_fn_base",
    # "simple_inventory_based_reward",
    # "simple_stat_kill_entity_based_reward",
    # "complex_inventory_based_reward",
    "inventory_based_reward",
    "possess_item_reward",
    "survive_per_day_reward",
    "survive_n_days_reward",
    "use_any_item_reward",
    "nearby_has_item_reward",
    "pos_change_reward"
]


# takes an initial info dict (t = 0), a pre info dict (t - 1), a current info dict (t), and elapsed time-steps,
# return a scalar reward value
reward_fn_base = Callable[
    [
        Arg(dict, "ini_info_dict"),
        Arg(dict, "pre_info_dict"),
        Arg(dict, "cur_info_dict"),
        Arg(int, "elapsed_timesteps"),
    ],
    float,
]


def _simple_stat_kill_entity_based_reward(
    name: str,
    weight: Union[int, float],
    ini_info_dict: dict,
    pre_info_dict: dict,
    cur_info_dict: dict,
    elapsed_timesteps: int,
):
    """
    A simple reward based on increment in `info["stat"]["kill_entity"][{name}]`.
    """
    return weight * (
        cur_info_dict["stat"]["kill_entity"].get(name, 0)
        - pre_info_dict["stat"]["kill_entity"].get(name, 0)
    )


def simple_stat_kill_entity_based_reward(
    name: str, weight: Union[int, float], **kwargs
) -> reward_fn_base:
    return partial(_simple_stat_kill_entity_based_reward, name=name, weight=weight)


def _simple_inventory_based_reward(
    name: str,
    weight: Union[int, float],
    ini_info_dict: dict,
    pre_info_dict: dict,
    cur_info_dict: dict,
    elapsed_timesteps: int,
):
    """
    A simple reward based on increment in `info["inventory"]`
    """
    return (
        sum(
            [
                inv_item["quantity"]
                for inv_item in cur_info_dict["inventory"]
                if inv_item["name"] == name
            ]
        )
        - sum(
            [
                inv_item["quantity"]
                for inv_item in pre_info_dict["inventory"]
                if inv_item["name"] == name
            ]
        )
    ) * weight


def simple_inventory_based_reward(
    name: str, weight: Union[int, float], **kwargs
) -> reward_fn_base:
    return partial(_simple_inventory_based_reward, name=name, weight=weight)

def _complex_inventory_based_reward(
    name: str,
    weight: Union[int, float],
    ini_info_dict: dict,
    pre_info_dict: dict,
    cur_info_dict: dict,
    elapsed_timesteps: int,
):
    """
    A simple reward based on increment in `info["inventory"]`
    """
    return (
        sum(
            [
                inv_item["quantity"]
                for inv_item in cur_info_dict["inventory"]
                if names_dict[inv_item["name"]][inv_item["variant"]] == name
            ]
        )
        - sum(
            [
                inv_item["quantity"]
                for inv_item in pre_info_dict["inventory"]
                if names_dict[inv_item["name"]][inv_item["variant"]] == name
            ]
        )
    ) * weight


def complex_inventory_based_reward(
    name: str, weight: Union[int, float], **kwargs
) -> reward_fn_base:
    return partial(_complex_inventory_based_reward, name=name, weight=weight)

def _inventory_based_reward(
    name: str,
    weight: Union[int, float],
    ini_info_dict: dict,
    pre_info_dict: dict,
    cur_info_dict: dict,
    elapsed_timesteps: int,
):
    """
    A simple reward based on increment in `info["inventory"]`
    """
    return (
        sum(
            [
                inv_item["quantity"]
                for inv_item in cur_info_dict["inventory"]
                if names_dict[inv_item["name"]][inv_item["variant"]] == name or inv_item["name"] == name
            ]
        )
        - sum(
            [
                inv_item["quantity"]
                for inv_item in pre_info_dict["inventory"]
                if names_dict[inv_item["name"]][inv_item["variant"]] == name or inv_item["name"] == name
            ]
        )
    ) * weight


def inventory_based_reward(
    name: str, weight: Union[int, float], **kwargs
) -> reward_fn_base:
    return partial(_inventory_based_reward, name=name, weight=weight)



def _possess_item_reward(
    name: str,
    weight: Union[int, float],
    quantity: int,
    ini_info_dict: dict,
    pre_info_dict: dict,
    cur_info_dict: dict,
    elapsed_timesteps: int,
):
    return (
        float(
            (
                sum(
                    [
                        inv_item["quantity"]
                        for inv_item in cur_info_dict["inventory"]
                        if inv_item["name"] == name
                    ]
                )
                - sum(
                    [
                        inv_item["quantity"]
                        for inv_item in ini_info_dict["inventory"]
                        if inv_item["name"] == name
                    ]
                )
            )
            >= quantity
        )
        * weight
    )


def possess_item_reward(
    name: str, quantity: int, weight: Union[int, float]
) -> reward_fn_base:
    return partial(_possess_item_reward, name=name, quantity=quantity, weight=weight)


def _survive_per_day_reward(
    mc_ticks_per_day: int,
    weight: Union[int, float],
    ini_info_dict: dict,
    pre_info_dict: dict,
    cur_info_dict: dict,
    elapsed_timesteps: int,
):
    time_since_death_pre = pre_info_dict["stat"]["time_since_death"]
    time_since_death_cur = cur_info_dict["stat"]["time_since_death"]
    survived_days_pre = time_since_death_pre // mc_ticks_per_day
    survived_days_cur = time_since_death_cur // mc_ticks_per_day
    return (survived_days_cur - survived_days_pre) * weight


def survive_per_day_reward(
    mc_ticks_per_day: int, weight: Union[int, float]
) -> reward_fn_base:
    return partial(
        _survive_per_day_reward, mc_ticks_per_day=mc_ticks_per_day, weight=weight
    )


def _survive_n_days_reward(
    mc_ticks_per_day: int,
    target_days: int,
    weight: Union[int, float],
    ini_info_dict: dict,
    pre_info_dict: dict,
    cur_info_dict: dict,
    elapsed_timesteps: int,
):
    return weight * float(
        cur_info_dict["stat"]["time_since_death"] >= mc_ticks_per_day * target_days
    )


def survive_n_days_reward(
    mc_ticks_per_day: int, target_days: int, weight: Union[int, float]
) -> reward_fn_base:
    return partial(
        _survive_n_days_reward,
        mc_ticks_per_day=mc_ticks_per_day,
        target_days=target_days,
        weight=weight,
    )


def _use_any_item_reward(
    items_and_weights: Dict[str, Union[int, float]],
    ini_info_dict: dict,
    pre_info_dict: dict,
    cur_info_dict: dict,
    elapsed_timesteps: int,
):
    """
    reward any usage of item in items_and_weights.keys()
    """
    return sum(
        [
            (
                cur_info_dict["stat"]["use_item"]["minecraft"][item]
                - pre_info_dict["stat"]["use_item"]["minecraft"][item]
            )
            * weight
            for item, weight in items_and_weights.items()
        ]
    )


def use_any_item_reward(
    items_and_weights: Dict[str, Union[int, float]]
) -> reward_fn_base:
    return partial(_use_any_item_reward, items_and_weights=items_and_weights)

def _nearby_has_item_reward(
    items_and_weights: Dict[str, Union[int, float]],
    ini_info_dict: dict,
    pre_info_dict: dict,
    cur_info_dict: dict,
    elapsed_timesteps: int,
):
    """
    reward any nearby items
    """
    return sum(
        [
            weight if item in cur_info_dict["voxels"]["block_name"] else 0.0
            for item, weight in items_and_weights.items()
        ]
    )

def nearby_has_item_reward(
    items_and_weights: Dict[str, Union[int, float]]
) -> reward_fn_base:
    return partial(_nearby_has_item_reward, items_and_weights = items_and_weights)

def _pos_change_reward(
    pos_and_weights: Dict[str, Tuple[str, float, float]],
    ini_info_dict: dict,
    pre_info_dict: dict,
    cur_info_dict: dict,
    elapsed_timesteps: int,
):
    """
    reward position change
    """
    reward = 0.0
    for pos, (dir, best_val, weight) in pos_and_weights.items():
        if dir == "+":
            if cur_info_dict[pos] > best_val:
                reward += (cur_info_dict[pos] - best_val) * weight
                pos_and_weights[pos] = (dir, cur_info_dict[pos], weight)
        elif dir == "-":
            if cur_info_dict[pos] < best_val:
                reward += (best_val - cur_info_dict[pos]) * weight
                pos_and_weights[pos] = (dir, cur_info_dict[pos], weight)
    return reward

def pos_change_reward(
    pos_and_weights: Dict[str, Tuple[str, float, float]]
) -> reward_fn_base:
    return partial(_pos_change_reward, pos_and_weights = pos_and_weights)
