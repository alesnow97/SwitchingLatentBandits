import math
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t
import json

import utils

plt.style.use('seaborn-v0_8-bright')

styles = ['Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid', 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark', 'seaborn-dark-palette', 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid', 'tableau-colorblind10']

sns.set_context(rc={"font.family": 'sans',
                    "font.size": 12,
                    "axes.titlesize": 25,
                    "axes.labelsize": 24,
                    "ytick.labelsize": 15,
                    "xtick.labelsize": 15,
                    "lines.linewidth": 3,
                    })

def ci2(mean, std, n, conf=0.025):
    # Calculate the t-value
    t_value = t.ppf(1 - conf, n - 1)

    # Calculate the margin of error
    margin_error = t_value * std / math.sqrt(n)

    # Calculate the lower and upper bounds of the confidence interval
    lower_bound = mean - margin_error
    upper_bound = mean + margin_error
    return lower_bound, upper_bound


if __name__ == '__main__':

    base_path = utils.get_base_path()

    algorithms_to_use = ['sliding_w_UCB', 'epsilon_greedy', 'exp3S',
                         'our_policy']
    paths_to_read_from = ['experiments/regret_figure_2a/3states_4actions_5obs/bandit11/exp12',
                          'experiments/movielens_figure_2b/5states_18actions_5obs/bandit8/exp0']

    paths_to_read_from = [os.path.join(base_path, i) for i in paths_to_read_from]

    fig, axs = plt.subplots(1, len(paths_to_read_from), figsize=(20, 6))  # , sharex=True, sharey=True)
    num_experiments = 5
    exp_data = []

    for i, path in enumerate(paths_to_read_from):
        # Opening JSON file
        f = open(path + '/exp_info.json')
        # returns JSON object as
        # a dictionary
        data = json.load(f)
        oracle_list = np.array(data['rewards']['oracle'])
        sliding_w_UCB_list = np.array(data['rewards']['sliding_w_UCB'])
        epsilon_greedy_list = np.array(data['rewards']['epsilon_greedy'])
        exp3S_list = np.array(data['rewards']['exp3S'])
        particle_filter_list = np.array(data['rewards']['particle_filter'])
        our_policy_list = np.array(data['rewards']['our_policy'])

        oracle_rewards = oracle_list[:, :, 1]
        x_axis = [i for i in range(oracle_rewards.shape[1])]

        sliding_w_UCB_regrets = oracle_rewards - sliding_w_UCB_list[:, :, 1]
        cumulative_regrets = np.cumsum(sliding_w_UCB_regrets, axis=1)
        cumulative_regret_mean = np.mean(cumulative_regrets, axis=0)
        cumulative_regret_std = np.std(cumulative_regrets, axis=0)
        lower_bound, upper_bound = ci2(cumulative_regret_mean,
                                       cumulative_regret_std, num_experiments)
        axs[i].plot(cumulative_regret_mean, 'c', label='SW-UCB')
        axs[i].fill_between(x_axis, lower_bound, upper_bound,
                                color='c', alpha=.2)

        epsilon_greedy_regrets = oracle_rewards - epsilon_greedy_list[:, :, 1]
        cumulative_regrets = np.cumsum(epsilon_greedy_regrets, axis=1)
        cumulative_regret_mean = np.mean(cumulative_regrets, axis=0)
        cumulative_regret_std = np.std(cumulative_regrets, axis=0)
        lower_bound, upper_bound = ci2(cumulative_regret_mean,
                                       cumulative_regret_std, num_experiments)
        axs[i].plot(cumulative_regret_mean, 'b', label='$\epsilon$-gr')
        axs[i].fill_between(x_axis, lower_bound, upper_bound,
                                color='b', alpha=.2)

        exp3S_regrets = oracle_rewards - exp3S_list[:, :, 1]
        cumulative_regrets = np.cumsum(exp3S_regrets, axis=1)
        cumulative_regret_mean = np.mean(cumulative_regrets, axis=0)
        cumulative_regret_std = np.std(cumulative_regrets, axis=0)
        # print(f"Exp3S regret {exp3S_regret.sum()}")
        lower_bound, upper_bound = ci2(cumulative_regret_mean,
                                       cumulative_regret_std, num_experiments)
        axs[i].plot(cumulative_regret_mean, 'g', label='Exp3.S')
        axs[i].fill_between(x_axis, lower_bound, upper_bound,
                                color='g', alpha=.2)

        particle_filter_regrets = oracle_rewards - particle_filter_list[:, :, 1]
        cumulative_regrets = np.cumsum(particle_filter_regrets, axis=1)
        cumulative_regret_mean = np.mean(cumulative_regrets, axis=0)
        cumulative_regret_std = np.std(cumulative_regrets, axis=0)
        lower_bound, upper_bound = ci2(cumulative_regret_mean,
                                       cumulative_regret_std,
                                       num_experiments)
        # print(f"particle_filter regret {exp3S_regret.sum()}")
        axs[i].plot(cumulative_regret_mean, 'y', label='PF')
        axs[i].fill_between(x_axis, lower_bound, upper_bound,
                            color='y', alpha=.2)

        our_policy_regrets = oracle_rewards - our_policy_list[:, :, 1]
        cumulative_regrets = np.cumsum(our_policy_regrets, axis=1)
        cumulative_regret_mean = np.mean(cumulative_regrets, axis=0)
        cumulative_regret_std = np.std(cumulative_regrets, axis=0)
        print(f"Our policy regret {our_policy_regrets.sum()}")
        lower_bound, upper_bound = ci2(cumulative_regret_mean,
                                       cumulative_regret_std, num_experiments)
        axs[i].plot(cumulative_regret_mean, 'r', label='SL-EC')
        axs[i].fill_between(x_axis, lower_bound, upper_bound,
                                color='r', alpha=.2)

        if i == 0:
            axs[i].set_title('(a)')
        else:
            axs[i].set_title('(b)')
        axs[i].spines['left'].set_linewidth(2)
        axs[i].spines['bottom'].set_linewidth(2)
        axs[i].spines['top'].set_linewidth(1)
        axs[i].spines['right'].set_linewidth(1)

        axs[i].spines['left'].set_capstyle('butt')
        axs[i].spines['bottom'].set_capstyle('butt')
        axs[i].spines['top'].set_capstyle('butt')
        axs[i].spines['right'].set_capstyle('butt')
        # axs[i].ticklabel_format(useOffset=True)
        axs[i].set_xlabel('t')
        axs[i].set_ylabel('$\widehat{\mathfrak{R}}(t)$')
        axs[i].grid()
        if i == 0:
            axs[i].legend(prop={'size': 15})

    plt.tight_layout()
    plt.show()
