import json
import os

import numpy as np

import utils
from simulations.regret_movielens_figure_2b import \
    RegretMovielensFigure2b
from environment.switchingBanditEnv import SwitchingBanditEnv


def create_switching_bandit_env():
    base_path = utils.get_base_path()
    file_path = 'transformed_data.json'
    data_path = os.path.join(base_path, "movielens_dataset", file_path)

    with open(data_path, 'r') as file:
        loaded_dictionary = json.load(file)
    transition_matrix = np.array(loaded_dictionary["transition_matrix"])
    user_genre_rating = np.array(loaded_dictionary["state_action_obs"])
    switching_environment = SwitchingBanditEnv(
        num_states=5, num_actions=18, num_obs=5,
        transition_matrix=transition_matrix,
        state_action_reward_matrix=user_genre_rating,
        possible_rewards=np.array([0.2, 0.4, 0.6, 0.8, 1.0]),
        is_movielens=True
    )
    print(f"sigma min is {switching_environment.sigma_min}")

    return switching_environment


if __name__ == '__main__':
    num_states = 3
    num_actions = 4
    num_obs = 5
    total_horizon = 10000
    delta = 0.99

    set_min_transition_prob = True
    if set_min_transition_prob:
        min_transition_prob = 0.3
    else:
        min_transition_prob = None

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

    switching_env = create_switching_bandit_env()


    num_particles = 300

    experiments = RegretMovielensFigure2b(
          switching_env=switching_env,
          num_experiments=num_experiments,
          sliding_window_size=1000,
          epsilon=0.15,
          exp3S_gamma=0.01,
          exp3S_limit=20,
          exp3S_normalization_factor=100,
          save_results=save_results,
          save_bandit_info=save_bandit_info,
          loaded_bandit=load_files,
          num_particles=num_particles,
          lowest_prob=10**(-4),
          num_lowest_prob=num_particles/4)

    experiments.run(total_horizon=total_horizon,
                    delta=delta,
                    compute_regret_exploitation_horizon=False)



