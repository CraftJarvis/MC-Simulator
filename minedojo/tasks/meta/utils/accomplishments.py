from functools import partial
from mypy_extensions import Arg
from typing import Union, Callable, Dict, Tuple, List
import json
import os 

# TODO: integrate the three accomplishments into one function!

__all__ = [
    "accomplishment_fn_base",
    # "simple_inventory_based_accomplishments",
    # "simple_kill_entity_based_accomplishments",
    # "complex_inventory_based_accomplishments",
    "inventory_based_accomplishments",
]

with open(os.path.join(os.path.dirname(__file__), "names.json"),"r") as f:
    names_dict = json.load(f)

accomplishment_fn_base = Callable[
    [
        Arg(dict, "ini_info_dict"),
        Arg(dict, "pre_info_dict"),
        Arg(dict, "cur_info_dict"),
        Arg(int, "elapsed_timesteps"),
    ],
    list,
]

def _inventory_based_accomplishments(
    names: List[str], # this is the goals set defined in the environment 
    # names: dict, # for multiple items 
    ini_info_dict: dict,
    pre_info_dict: dict,
    cur_info_dict: dict,
    elapsed_timesteps: int,
):
    """
    A reward based on increment in `info["inventory"]`
    The accompolishments are item-centric, i.e., checking the dropping items of goals
    for two types: 
        1. item ids: like "stone", "sand", w/o variants
        2. item Names: like "Stone", "Dirt", w/ variants of id
    """
    counts = dict()

    # For id check 
    for inv_item in cur_info_dict["inventory"]:
        inv_item_name = inv_item["name"]
        if inv_item_name in names:
            if inv_item_name in counts.keys():
                counts[inv_item_name] += inv_item["quantity"]
            else:
                counts[inv_item_name] = inv_item["quantity"]

    for inv_item in pre_info_dict["inventory"]:
        inv_item_name = inv_item["name"]
        if inv_item_name in names:
            if inv_item_name in counts.keys():
                counts[inv_item_name] -= inv_item["quantity"]
            else:
                counts[inv_item_name] = -inv_item["quantity"]


    # for name check 
    for inv_item in cur_info_dict["inventory"]:
        inv_item_name = names_dict[inv_item["name"]][inv_item["variant"]]
        if inv_item_name in names:
            if inv_item_name in counts.keys():
                counts[inv_item_name] += inv_item["quantity"]
            else:
                counts[inv_item_name] = inv_item["quantity"]

    for inv_item in pre_info_dict["inventory"]:
        inv_item_name = names_dict[inv_item["name"]][inv_item["variant"]]
        if inv_item_name in names:
            if inv_item_name in counts.keys():
                counts[inv_item_name] -= inv_item["quantity"]
            else:
                counts[inv_item_name] = -inv_item["quantity"]

    # above process counts the inventory difference

    accomplishments = []
    for name, quantity_diff in counts.items():
        if quantity_diff > 0:
            accomplishments.append(name)

    return accomplishments


def _simple_inventory_based_accomplishments(
    names: List[str],
    ini_info_dict: dict,
    pre_info_dict: dict,
    cur_info_dict: dict,
    elapsed_timesteps: int,
):
    """
    A simple reward based on increment in `info["inventory"]`
    """
    counts = dict()
    for inv_item in cur_info_dict["inventory"]:
        if inv_item["name"] in names:
            if inv_item["name"] in counts.keys():
                counts[inv_item["name"]] += inv_item["quantity"]
            else:
                counts[inv_item["name"]] = inv_item["quantity"]

    for inv_item in pre_info_dict["inventory"]:
        if inv_item["name"] in names:
            if inv_item["name"] in counts.keys():
                counts[inv_item["name"]] -= inv_item["quantity"]
            else:
                counts[inv_item["name"]] = -inv_item["quantity"]

    accomplishments = []
    for name, quantity_diff in counts.items():
        if quantity_diff > 0:
            accomplishments.append(name)

    return accomplishments

def _complex_inventory_based_accomplishments(
    names: List[str],
    ini_info_dict: dict,
    pre_info_dict: dict,
    cur_info_dict: dict,
    elapsed_timesteps: int,
):
    """
    A simple reward based on increment in `info["inventory"]`
    """
    counts = dict()
    for inv_item in cur_info_dict["inventory"]:
        inv_item_name = names_dict[inv_item["name"]][inv_item["variant"]]
        if inv_item_name in names:
            if inv_item_name in counts.keys():
                counts[inv_item_name] += inv_item["quantity"]
            else:
                counts[inv_item_name] = inv_item["quantity"]

    for inv_item in pre_info_dict["inventory"]:
        inv_item_name = names_dict[inv_item["name"]][inv_item["variant"]]
        if inv_item_name in names:
            if inv_item_name in counts.keys():
                counts[inv_item_name] -= inv_item["quantity"]
            else:
                counts[inv_item_name] = -inv_item["quantity"]

    accomplishments = []
    for name, quantity_diff in counts.items():
        if quantity_diff > 0:
            accomplishments.append(name)

    return accomplishments

def _simple_kill_entity_based_accomplishments(
    names: List[str],
    ini_info_dict: dict,
    pre_info_dict: dict,
    cur_info_dict: dict,
    elapsed_timesteps: int,
):
    accomplishments = []
    for name in cur_info_dict["stat"]["kill_entity"].keys():
        if cur_info_dict["stat"]["kill_entity"][name] - pre_info_dict["stat"]["kill_entity"][name] > 0:
            accomplishments.append(name)
    
    return accomplishments

def inventory_based_accomplishments(
    names: List[str], **kwargs
)-> accomplishment_fn_base:
    return partial(_inventory_based_accomplishments, names=names)

def simple_inventory_based_accomplishments(
    names: List[str], **kwargs
) -> accomplishment_fn_base:
    return partial(_simple_inventory_based_accomplishments, names=names)

def complex_inventory_based_accomplishments(
    names: List[str], **kwargs
) -> accomplishment_fn_base:
    return partial(_complex_inventory_based_accomplishments, names=names)

def simple_kill_entity_based_accomplishments(
    names: List[str], **kwargs
) -> accomplishment_fn_base:
    return partial(_simple_kill_entity_based_accomplishments, names=names)