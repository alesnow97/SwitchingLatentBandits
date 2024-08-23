import math
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t
import json

import utils

plt.style.use('seaborn-v0_8-bright')

# visualization library
sns.set_context(rc={"font.family": 'sans',
                    "font.size": 12,
                    "axes.titlesize": 25,
                    "axes.labelsize": 25,
                    "ytick.labelsize": 20,
                    "xtick.labelsize": 20,
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

    # EXPERIMENT WITH 3 STATES, 4 ACTIONS AND 4 OBSERVATIONS
    base_path = utils.get_base_path()

    paths_to_read_from_first_exp = [
        'experiments/misspecified_figure_5/3states_4actions_4obs/bandit0/exp0_mis0.0',
        'experiments/misspecified_figure_5/3states_4actions_4obs/bandit0/exp1_mis0.01',
        'experiments/misspecified_figure_5/3states_4actions_4obs/bandit0/exp2_mis0.02',
        'experiments/misspecified_figure_5/3states_4actions_4obs/bandit0/exp3_mis0.03',
        'experiments/misspecified_figure_5/3states_4actions_4obs/bandit0/exp4_mis0.04',
        'experiments/misspecified_figure_5/3states_4actions_4obs/bandit0/exp5_mis0.05',
        'experiments/misspecified_figure_5/3states_4actions_4obs/bandit0/exp6_mis0.07',
        'experiments/misspecified_figure_5/3states_4actions_4obs/bandit0/exp7_mis0.1',
        ]

    # EXPERIMENT WITH 5 STATES, 5 ACTIONS AND 5 OBSERVATIONS
    paths_to_read_from_second_exp = [
        'experiments/misspecified_figure_5/5states_5actions_5obs/bandit0/exp0_mis0.0',
        'experiments/misspecified_figure_5/5states_5actions_5obs/bandit0/exp3_mis0.01',
        'experiments/misspecified_figure_5/5states_5actions_5obs/bandit0/exp1_mis0.02',
        'experiments/misspecified_figure_5/5states_5actions_5obs/bandit0/exp4_mis0.03',
        'experiments/misspecified_figure_5/5states_5actions_5obs/bandit0/exp5_mis0.04',
        'experiments/misspecified_figure_5/5states_5actions_5obs/bandit0/exp2_mis0.05',
        'experiments/misspecified_figure_5/5states_5actions_5obs/bandit0/exp7_mis0.07',
        'experiments/misspecified_figure_5/5states_5actions_5obs/bandit0/exp6_mis0.1',
        ]

    paths_to_read_from_first_exp = [os.path.join(base_path, i) for i in paths_to_read_from_first_exp]
    paths_to_read_from_second_exp = [os.path.join(base_path, i) for i in paths_to_read_from_second_exp]

    fig, axs = plt.subplots(1, 2,
                            figsize=(25, 10))
    x_axis = None
    exp_data = []
    num_experiments = 10

    line_plot_colors1 = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                         'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    misspecification_level = ['0.0', '0.01', '0.02', '0.03', '0.04', '0.05', '0.07', '0.1', '0.2']

    starting_index = 1

    for j, path in enumerate(paths_to_read_from_first_exp):
        # Opening JSON file
        entire_path = path + '/exp_info.json'
        f = open(entire_path)
        # returns JSON object as
        # a dictionary

        data = json.load(f)
        estimation_errors = np.array(data['estimation_errors'])
        num_states = data['num_states']
        if x_axis is None:
            x_axis = estimation_errors[0, starting_index:, 0]

        # version with Frobenious norm
        estimation_errors = estimation_errors[:, starting_index:, 1].reshape(num_experiments, -1)

        mean_estimation_errors = np.mean(estimation_errors, axis=0)
        std_estimation_errors = np.std(estimation_errors, axis=0, ddof=1)
        lower_bound, upper_bound = ci2(mean_estimation_errors, std_estimation_errors, num_experiments)
        axs[0].plot(x_axis, mean_estimation_errors, line_plot_colors1[j], label=f'm={misspecification_level[j]}')
        axs[0].fill_between(x_axis, lower_bound, upper_bound, color=line_plot_colors1[j], alpha=0.2)

    axs[0].set_title('3 States, 4 Actions, 4 Observations', weight='bold')
    axs[0].spines['left'].set_linewidth(2)
    axs[0].spines['bottom'].set_linewidth(2)
    axs[0].spines['top'].set_linewidth(1)
    axs[0].spines['right'].set_linewidth(1)

    axs[0].spines['left'].set_capstyle('butt')
    axs[0].spines['bottom'].set_capstyle('butt')
    axs[0].spines['top'].set_capstyle('butt')
    axs[0].spines['right'].set_capstyle('butt')
    axs[0].set_xlabel('t')
    axs[0].set_ylabel('Estimation Error', weight='bold')
    axs[0].grid()

    for j, path in enumerate(paths_to_read_from_second_exp):
        # Opening JSON file
        entire_path = path + '/exp_info.json'
        f = open(entire_path)
        # returns JSON object as
        # a dictionary

        data = json.load(f)
        estimation_errors = np.array(data['estimation_errors'])
        num_states = data['num_states']
        if x_axis is None:
            x_axis = estimation_errors[0, starting_index:, 0]

        # version with Frobenious norm
        estimation_errors = estimation_errors[:, starting_index:, 1].reshape(
            num_experiments, -1)

        mean_estimation_errors = np.mean(estimation_errors, axis=0)
        std_estimation_errors = np.std(estimation_errors, axis=0, ddof=1)
        lower_bound, upper_bound = ci2(mean_estimation_errors,
                                       std_estimation_errors, num_experiments)
        axs[1].plot(x_axis, mean_estimation_errors, line_plot_colors1[j],
                    label=f'm={misspecification_level[j]}')
        axs[1].fill_between(x_axis, lower_bound, upper_bound,
                            color=line_plot_colors1[j], alpha=0.2)

    axs[1].set_title('5 States, 5 Actions, 5 Observations', weight='bold')
    axs[1].spines['left'].set_linewidth(2)
    axs[1].spines['bottom'].set_linewidth(2)
    axs[1].spines['top'].set_linewidth(1)
    axs[1].spines['right'].set_linewidth(1)

    axs[1].spines['left'].set_capstyle('butt')
    axs[1].spines['bottom'].set_capstyle('butt')
    axs[1].spines['top'].set_capstyle('butt')
    axs[1].spines['right'].set_capstyle('butt')
    axs[1].set_xlabel('t')
    axs[1].grid()

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 25})
    plt.tight_layout()
    plt.show()