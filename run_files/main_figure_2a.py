import os

import numpy as np

import utils
from simulations.regret_figure_2a import RegretFigure2a
from environment.switchingBanditEnv import SwitchingBanditEnv


if __name__ == '__main__':
    num_states = 3
    num_actions = 4
    num_obs = 5
    total_horizon = 10000
    delta = 0.99

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


    transition_from_file = None
    reference_matrix_from_file = None
    state_action_reward_matrix_from_file = None
    possible_rewards = None
    observation_multiplier = 20
    # transition_multiplier = 15
    transition_multiplier = 20
    num_experiments = 2

    bandit_to_load_path = os.path.join(
        utils.get_base_path(),
        "experiments/regret_figure_2a",
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
            utils.load_files(bandit_to_load_path, bandit_num, False)

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

    exploration_horizon = utils.find_optimal_exploration_length(
        switching,
        total_horizon,
        delta,
    )

    print(f"Exploration horizon is {exploration_horizon}")

    num_particles = 200
    dirichlet_prior = (switching.transition_matrix * 10).astype(int)
    dirichlet_prior = np.clip(dirichlet_prior, a_min=1, a_max=None)
    # dirichlet_prior = (switching.transition_matrix * 1).astype(int)
    experiments = RegretFigure2a(switching,
                                 num_experiments=num_experiments,
                                 sliding_window_size=1500,
                                 epsilon=0.1,
                                 exp3S_gamma=0.01,
                                 exp3S_limit=20,
                                 exp3S_normalization_factor=100,
                                 save_results=save_results,
                                 save_bandit_info=save_bandit_info,
                                 loaded_bandit=load_files,
                                 bandit_num=bandit_num,
                                 num_particles=num_particles,
                                 lowest_prob=10**(-4),
                                 num_lowest_prob=num_particles/4,
                                 dirichlet_prior=dirichlet_prior)

    experiments.run(total_horizon=total_horizon,
                    exploration_horizon=exploration_horizon,
                    compute_regret_exploitation_horizon=False)



