import os

import utils
from simulations.estimation_error_figure_1b import EstimationErrorFigure1b

from environment.switchingBanditEnv import SwitchingBanditEnv


if __name__ == '__main__':
    num_states = 3
    num_actions = 4
    num_obs = 5

    # starting_checkpoint = 20000
    # num_checkpoints = 51
    # checkpoint_duration = 1000

    starting_checkpoint = 2000
    num_checkpoints = 11
    checkpoint_duration = 1000

    new_run = False

    set_min_transition_prob = True
    if set_min_transition_prob:
        min_transition_prob = 0.3
    else:
        min_transition_prob = None

    if new_run is True:
        load_files = False
        save_results = True
        run_experiments = True
        save_bandit_info = True
    else:
        load_files = True
        save_results = True
        run_experiments = True
        save_bandit_info = False

    estimation_error_expV2 = True
    different_selected_arms = False

    transition_from_file = None
    reference_matrix_from_file = None
    state_action_reward_matrix_from_file = None
    possible_rewards = None
    observation_multiplier = 20
    # transition_multiplier = 15
    transition_multiplier = 20
    num_experiments = 20

    # bandit_to_load_path = 'experiments_estimation_error_v2/5states_8actions_10obs/'
    # bandit_to_load_path = 'experiments/3states_4actions_5obs/'
    bandit_to_load_path = os.path.join(
        utils.get_base_path(),
        "experiments/estimation_error_figure_1b",
        f"{num_states}states_{num_actions}actions_{num_obs}obs")

    # this variable represents the bandit instance representing the environment
    # to be learned. This variable is useful only when load_files is True
    bandit_num = 0

    num_bandits = 10
    num_arms_to_use = 5

    if load_files:
        num_states, num_actions, num_obs, \
            transition_from_file, reference_matrix_from_file, \
            state_action_reward_matrix_from_file, possible_rewards = \
            utils.load_files(bandit_to_load_path, bandit_num, True)

    switching = SwitchingBanditEnv(num_states=num_states,
                                   num_actions=num_actions,
                                   num_obs=num_obs,
                                   transition_matrix=transition_from_file,
                                   state_action_reward_matrix=state_action_reward_matrix_from_file,
                                   reference_matrix=reference_matrix_from_file,
                                   possible_rewards=possible_rewards,
                                   transition_multiplier=transition_multiplier,
                                   observation_multiplier=observation_multiplier,
                                   min_transition_prob=min_transition_prob)

    experiments = EstimationErrorFigure1b(switching,
                                       num_experiments=num_experiments,
                                       save_results=save_results,
                                       save_bandit_info=save_bandit_info,
                                       loaded_bandit=load_files,
                                       bandit_num=bandit_num)

    experiments.run(starting_checkpoint=starting_checkpoint,
                    checkpoint_duration=checkpoint_duration,
                    num_checkpoints=num_checkpoints)




