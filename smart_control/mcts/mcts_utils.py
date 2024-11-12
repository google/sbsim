
"""
    This returns all possible actions given water and air temperature ranges and steps.
    The number of actions returned by this method is the branching factor in the MCTS tree.
"""
def get_available_actions(t_water_low, t_water_high, t_air_low, t_air_high, water_step, air_step):
    possible_actions = set()
    for water_temp in range(t_water_low, t_water_high + water_step, water_step):
        for air_temp in range(t_air_low, t_air_high + air_step, air_step):
            possible_actions.add((water_temp, air_temp))
    return possible_actions
