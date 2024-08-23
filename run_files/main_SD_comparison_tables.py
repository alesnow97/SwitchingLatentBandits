import json

import numpy as np

import utils
from simulations.SD_comparison_tables import SDComparisonTable1
from proposed_simulations.estimation_error_v2 import EstimationErrorExpV2
from proposed_simulations.estimation_error_v2_rebuttal import \
    EstimationErrorExpV2Rebuttal
from environment.switchingBanditEnv import SwitchingBanditEnv


if __name__ == '__main__':
    num_states = 3
    num_actions = 4
    num_obs = 5
    total_horizon = 10000
    delta = 0.99

    theoretical = False
    theoretical_reduced = False

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

    compare_algo = False
    compare_algo_new = True
    compare_algo_movielens = False
    switching_movielens = False
    estimation_error_exp = False

    hmm = False
    estimation_error_exp_rebuttal = False
    estimation_error_exp_all_arms = False
    estimation_error_expV2_rebuttal = False

    # if hmm:
    #     estimation_error_exp_rebuttal = True
    #     estimation_error_expV2_rebuttal = False
    # else:
    #     estimation_error_exp_rebuttal = False
    #     estimation_error_expV2_rebuttal = True

    estimation_error_expV2 = False
    different_selected_arms = False


    transition_from_file = None
    reference_matrix_from_file = None
    state_action_reward_matrix_from_file = None
    possible_rewards = None
    observation_multiplier = 20
    # transition_multiplier = 15
    transition_multiplier = 20
    num_experiments = 20

    # 'experiments_estimation_error_v2/5states_8actions_10obs/bandit0/exp1'

    # bandit_to_load_path = 'experiments_estimation_error_v2/5states_8actions_10obs/'
    # bandit_to_load_path = 'experiments/3states_4actions_5obs/'
    bandit_to_load_path = f'experiments/{num_states}states_{num_actions}actions_{num_obs}obs/'
    bandit_num = 15

    experiments_samples = [9000, 1500000, 2100000, 3000000]
    num_bandits = 10
    num_arms_to_use = 5

    if load_files:
        num_states, num_actions, num_obs, \
            transition_from_file, reference_matrix_from_file, \
            state_action_reward_matrix_from_file, possible_rewards = \
            utils.load_files(bandit_to_load_path, bandit_num,
                             estimation_error_exp or estimation_error_expV2 or
                             estimation_error_exp_rebuttal or estimation_error_exp_all_arms)

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


    if estimation_error_exp_rebuttal:
        for i in range(num_bandits):
            if load_files:
                num_states, num_actions, num_obs, \
                    transition_from_file, reference_matrix_from_file, \
                    state_action_reward_matrix_from_file, possible_rewards = \
                    utils.load_files(bandit_to_load_path, i,
                                     estimation_error_expV2 or estimation_error_exp_rebuttal)
            switching = SwitchingBanditEnv(num_states=num_states,
                                           num_actions=num_actions,
                                           num_obs=num_obs,
                                           transition_matrix=transition_from_file,
                                           state_action_reward_matrix=state_action_reward_matrix_from_file,
                                           reference_matrix=reference_matrix_from_file,
                                           possible_rewards=possible_rewards,
                                           transition_multiplier=transition_multiplier,
                                           observation_multiplier=observation_multiplier)
            experiments = SDComparisonTable1(switching,
                                             save_results=save_results,
                                             save_bandit_info=save_bandit_info,
                                             loaded_bandit=load_files,
                                             bandit_num=i,
                                             dir_name=bandit_to_load_path,
                                             run_experiments=run_experiments)
            experiments.run(experiments_samples=experiments_samples)

    if estimation_error_expV2_rebuttal:
        for i in range(num_bandits):
            if load_files:
                num_states, num_actions, num_obs, \
                    transition_from_file, reference_matrix_from_file, \
                    state_action_reward_matrix_from_file, possible_rewards = \
                    utils.load_files(bandit_to_load_path, i,
                                     estimation_error_expV2_rebuttal or estimation_error_exp_rebuttal)
            switching = SwitchingBanditEnv(num_states=num_states,
                                           num_actions=num_actions,
                                           num_obs=num_obs,
                                           transition_matrix=transition_from_file,
                                           state_action_reward_matrix=state_action_reward_matrix_from_file,
                                           reference_matrix=reference_matrix_from_file,
                                           possible_rewards=possible_rewards,
                                           transition_multiplier=transition_multiplier,
                                           observation_multiplier=observation_multiplier)
            experiments = EstimationErrorExpV2Rebuttal(switching,
                                             num_experiments=num_experiments,
                                             save_results=save_results,
                                             save_bandit_info=save_bandit_info,
                                             loaded_bandit=load_files,
                                             dir_name=bandit_to_load_path,
                                             bandit_num=i)
            experiments.run(experiments_samples=experiments_samples, num_arms=num_arms_to_use)

    if estimation_error_expV2:
        experiments = EstimationErrorExpV2(switching,
                                           num_experiments=num_experiments,
                                           save_results=save_results,
                                           save_bandit_info=save_bandit_info,
                                           loaded_bandit=load_files,
                                           bandit_num=bandit_num)
        starting_checkpoint = 20000
        num_checkpoints = 51
        checkpoint_duration = 1000
        experiments.run(starting_checkpoint=starting_checkpoint,
                        checkpoint_duration=checkpoint_duration,
                        num_checkpoints=num_checkpoints)



