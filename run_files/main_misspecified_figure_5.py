import os

import utils
from simulations.estimation_error_misspecified_figure_5 import \
    EstimationErrorMisspecifiedFigure5
from environment.switchingBanditEnvMisspecified import SwitchingBanditEnvMisspecified

if __name__ == '__main__':
    num_states = 3
    num_actions = 4
    num_obs = 4

    starting_checkpoint = 2000
    num_checkpoints = 50
    checkpoint_duration = 2000

    new_run = True

    set_min_transition_prob = False
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

    transition_from_file = None
    reference_matrix_from_file = None
    state_action_reward_matrix_from_file = None
    possible_rewards = None
    observation_multiplier = 20
    # transition_multiplier = 15
    transition_multiplier = 20
    num_experiments = 10

    bandit_to_load_path = os.path.join(
        utils.get_base_path(),
        "experiments_misspecified_figure_5",
        f"{num_states}states_{num_actions}actions_{num_obs}obs")

    bandit_num = 0

    num_bandits = 10
    num_arms_to_use = 5

    misspecification_level = 0.07

    if load_files:
        num_states, num_actions, num_obs, \
            transition_from_file, real_reference_matrix_from_file, \
            state_action_reward_matrix_from_file, possible_rewards = \
            utils.load_files_misspecified(bandit_to_load_path, bandit_num)
    else:
        real_reference_matrix_from_file = None

    switching = SwitchingBanditEnvMisspecified(
       num_states=num_states,
       num_actions=num_actions,
       num_obs=num_obs,
       transition_matrix=transition_from_file,
       state_action_reward_matrix=state_action_reward_matrix_from_file,
       real_reference_matrix=real_reference_matrix_from_file,
       possible_rewards=possible_rewards,
       transition_multiplier=transition_multiplier,
       observation_multiplier=observation_multiplier,
       min_transition_prob=min_transition_prob,
       misspecification=misspecification_level)


    experiments = EstimationErrorMisspecifiedFigure5(
        switching,
        num_experiments=num_experiments,
        save_results=save_results,
        save_bandit_info=save_bandit_info,
        loaded_bandit=load_files,
        bandit_num=bandit_num)

    experiments.run(starting_checkpoint=starting_checkpoint,
                    checkpoint_duration=checkpoint_duration,
                    num_checkpoints=num_checkpoints)



