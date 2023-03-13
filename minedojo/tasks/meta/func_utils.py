from .utils import (
    always_satisfy_condition,
    # simple_inventory_based_check,
    # simple_stat_kill_entity_based_check,
    # complex_inventory_based_check,
    inventory_based_check,
    # simple_inventory_based_reward,
    # simple_stat_kill_entity_based_reward,
    # complex_inventory_based_reward,
    inventory_based_reward,
    nearby_has_item_reward,
    pos_change_reward,
    # simple_inventory_based_accomplishments,
    # simple_kill_entity_based_accomplishments,
    # complex_inventory_based_accomplishments,
    inventory_based_accomplishments,
    max_timesteps_check
)


def get_reward_fns(target_names, reward_weights):
    reward_fns = []
    if target_names is not None:
        # reward_fns += [
        #     simple_inventory_based_reward(name=k, weight=v)
        #     for k, v in reward_weights.items()
        # ]
        # reward_fns += [
        #     simple_stat_kill_entity_based_reward(name=k, weight=v)
        #     for k, v in reward_weights.items()
        # ]
        # reward_fns += [
        #     complex_inventory_based_reward(name=k, weight=v)
        #     for k, v in reward_weights.items()
        # ]
        reward_fns += [
            inventory_based_reward(name=k, weight=v)
            for k, v in reward_weights.items()
        ]
        
        

    return reward_fns


def get_success_criteria(target_names, target_quantities, max_nsteps):
    success_criteria = []
    if target_names is not None:
        # success_criteria += [
        #     simple_inventory_based_check(name=k, quantity=v)
        #     for k, v in target_quantities.items()
        # ]
        # success_criteria += [
        #     simple_stat_kill_entity_based_check(name=k, quantity=v)
        #     for k, v in target_quantities.items()
        # ]
        # success_criteria += [
        #     complex_inventory_based_check(name=k, quantity=v)
        #     for k, v in target_quantities.items()
        # ]
        success_criteria += [
            inventory_based_check(name=k, quantity=v)
            for k, v in target_quantities.items()
        ]
    if max_nsteps > 0:
        success_criteria += [
            max_timesteps_check(threshold = max_nsteps)
        ]

    return success_criteria