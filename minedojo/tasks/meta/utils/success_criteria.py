from functools import partial
from mypy_extensions import Arg
from typing import Callable, List, Dict, Union
import json
import os 

__all__ = [
    "check_success_base",
    # "simple_inventory_based_check",
    # "simple_stat_kill_entity_based_check",
    # "complex_inventory_based_check",
    "inventory_based_check",
    "time_since_death_check",
    "use_any_item_check",
    "use_all_item_check",
    "max_timesteps_check",
]

with open(os.path.join(os.path.dirname(__file__), "names.json"),"r") as f:
    names_dict = json.load(f)


# takes an initial info dict, a current info dict, and elapsed time-steps, return successful or not
check_success_base = Callable[
    [
        Arg(dict, "ini_info_dict"),
        Arg(dict, "cur_info_dict"),
        Arg(int, "elapsed_timesteps"),
    ],
    bool,
]


def _inventory_based_check(
    name: str,
    quantity: int,
    ini_info_dict: dict,
    cur_info_dict: dict,
    elapsed_timesteps: int,
):
    """
    Success check based on `info["inventory"]` both on id and names 
    """
    crit = (
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
                for inv_item in ini_info_dict["inventory"]
                if names_dict[inv_item["name"]][inv_item["variant"]] == name or inv_item["name"] == name
            ]
        )
    ) >= quantity
    return crit


def inventory_based_check(
    name: str, quantity: int, **kwargs
) -> check_success_base:
    return partial(_inventory_based_check, name=name, quantity=quantity)

def _simple_stat_kill_entity_based_check(
    name: str,
    quantity: int,
    ini_info_dict: dict,
    cur_info_dict: dict,
    elapsed_timesteps: int,
):
    """
    A simple success check based on `info["stat"]["kill_entity"][{name}]`.
    """
    return (
        cur_info_dict["stat"]["kill_entity"].get(name, 0)
        - ini_info_dict["stat"]["kill_entity"].get(name, 0)
    ) >= quantity


def simple_stat_kill_entity_based_check(
    name: str, quantity: int, **kwargs
) -> check_success_base:
    return partial(_simple_stat_kill_entity_based_check, name=name, quantity=quantity)

def _complex_inventory_based_check(
    name: str,
    quantity: int,
    ini_info_dict: dict,
    cur_info_dict: dict,
    elapsed_timesteps: int,
):
    """
    A simple success check based on `info["inventory"]`
    """
    crit = (
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
                for inv_item in ini_info_dict["inventory"]
                if names_dict[inv_item["name"]][inv_item["variant"]] == name
            ]
        )
    ) >= quantity
    return crit


def complex_inventory_based_check(
    name: str, quantity: int, **kwargs
) -> check_success_base:
    return partial(_complex_inventory_based_check, name=name, quantity=quantity)


def _simple_inventory_based_check(
    name: str,
    quantity: int,
    ini_info_dict: dict,
    cur_info_dict: dict,
    elapsed_timesteps: int,
):
    """
    A simple success check based on `info["inventory"]`
    """
    crit = (
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
    ) >= quantity
    return crit


def simple_inventory_based_check(
    name: str, quantity: int, **kwargs
) -> check_success_base:
    return partial(_simple_inventory_based_check, name=name, quantity=quantity)


def _time_since_death_check(
    threshold, ini_info_dict: dict, cur_info_dict: dict, elapsed_timesteps: int
):
    """
    Success check based on info["time_since_death"]
    """
    return cur_info_dict["stat"]["time_since_death"] >= threshold


def time_since_death_check(threshold, **kwargs) -> check_success_base:
    return partial(_time_since_death_check, threshold=threshold)


def _max_timesteps_check(
    threshold, ini_info_dict: dict, cur_info_dict: dict, elapsed_timesteps: int
):
    return elapsed_timesteps >= threshold


def max_timesteps_check(threshold, **kwargs) -> check_success_base:
    return partial(_max_timesteps_check, threshold=threshold)


def _use_any_item_check(
    targets: Dict[str, int],
    ini_info_dict: dict,
    cur_info_dict: dict,
    elapsed_timesteps: int,
):
    """
    success check based on increment in info["stat"]["use_item"]["minecraft"][item]
    satisfaction of any item will result in "True" -- the logic "any"
    """
    return any(
        [
            (
                cur_info_dict["stat"]["use_item"]["minecraft"][item]
                - ini_info_dict["stat"]["use_item"]["minecraft"][item]
            )
            >= target
            for item, target in targets.items()
        ]
    )


def use_any_item_check(targets: Dict[str, int]) -> check_success_base:
    return partial(_use_any_item_check, targets=targets)


def _use_all_item_check(
    targets: Dict[str, int],
    ini_info_dict: dict,
    cur_info_dict: dict,
    elapsed_timesteps: int,
):
    """
    success check based on increment in info["stat"]["use_item"]["minecraft"][item]
    satisfaction of all item will result in "True" -- the logic "all"
    """
    return all(
        [
            (
                cur_info_dict["stat"]["use_item"]["minecraft"][item]
                - ini_info_dict["stat"]["use_item"]["minecraft"][item]
            )
            >= target
            for item, target in targets.items()
        ]
    )


def use_all_item_check(targets: Dict[str, int]) -> check_success_base:
    return partial(_use_all_item_check, targets=targets)
