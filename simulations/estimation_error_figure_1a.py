import utils
from policies.our_policy_figure_1a import OurPolicyFigure1a
from environment.switchingBanditEnv import SwitchingBanditEnv
import numpy as np
import os
import json


class EstimationErrorFigure1a:

    def __init__(self, switching_env: SwitchingBanditEnv,
                 num_experiments, save_results, loaded_bandit,
                 save_bandit_info=False, bandit_num=0):
        self.switching_env = switching_env
        self.num_experiments = num_experiments
        self.bandit_dir_path = None
        self.new_bandit_index = None
        self.loaded_bandit = loaded_bandit
        self.bandit_num = bandit_num
        self.generate_dirs()
        self.save_results = save_results
        self.save_bandit_info = save_bandit_info

    def generate_dirs(self):
        utils.set_base_path()
        dir_name = f"experiments/estimation_error_figure_1a/{self.switching_env.num_states}states_{self.switching_env.num_actions}actions_{self.switching_env.num_obs}obs"
        if os.path.exists(dir_name):
            os.chdir(os.path.join(os.getcwd(), dir_name))
            if self.loaded_bandit:
                self.bandit_dir_path = os.path.join(os.getcwd(), f'bandit{self.bandit_num}')
                self.new_exp_index = len(os.listdir(self.bandit_dir_path)) - 1
            else:
                self.new_bandit_index = len(os.listdir(os.getcwd()))
                self.bandit_dir_path = os.path.join(os.getcwd(), f'bandit{self.new_bandit_index}')
                os.mkdir(self.bandit_dir_path)
                self.new_exp_index = 0
        else:
            splitted_dir_name = dir_name.split("/")
            first_part, second_part = splitted_dir_name[0], splitted_dir_name[1]
            new_dir_path = os.path.join(os.getcwd(), first_part)
            if not os.path.exists(new_dir_path):
                os.mkdir(new_dir_path)
            os.chdir(new_dir_path)
            new_subdir_path = os.path.join(os.getcwd(), second_part)
            os.mkdir(new_subdir_path)
            os.chdir(new_subdir_path)
            self.new_bandit_index = 0
            self.bandit_dir_path = os.path.join(os.getcwd(), f'bandit{self.new_bandit_index}')
            os.mkdir(self.bandit_dir_path)
            self.new_exp_index = 0
        self.exp_dir_path = os.path.join(self.bandit_dir_path, f'exp{self.new_exp_index}')
        os.mkdir(self.exp_dir_path)

    def run(self, starting_checkpoint, num_checkpoints, checkpoint_duration,
            num_selected_arms_list):
        transition_matrix = self.switching_env.transition_matrix
        state_action_reward_matrix = self.switching_env.state_action_reward_matrix

        print(f"Starting checkpoint is {starting_checkpoint}")

        bandit_info_dict = {
            'transition_matrix': self.switching_env.transition_matrix.tolist(),
            'reference_matrix': self.switching_env.reference_matrix.tolist(),
            'state_action_reward_matrix': self.switching_env.state_action_reward_matrix.tolist(),
            'num_states': self.switching_env.num_states,
            'num_actions': self.switching_env.num_actions,
            'num_obs': self.switching_env.num_obs}

        result_dict = {'starting_checkpoint': starting_checkpoint,
                       'num_checkpoints': num_checkpoints,
                       'checkpoint_duration': checkpoint_duration,
                       'num_experiments': self.num_experiments}

        self.num_checkpoints = num_checkpoints
        total_horizon = starting_checkpoint + (self.num_checkpoints - 1) * checkpoint_duration

        run_result = np.empty((len(num_selected_arms_list), self.num_experiments, self.num_checkpoints, 2))
        list_of_iteration_dict = []

        for j, num_selected_arms in enumerate(num_selected_arms_list):
            iteration_dict = {'num_selected_arms': num_selected_arms}

            estimation_results_list = np.empty((self.num_experiments, self.num_checkpoints, 2))
            our_policy = OurPolicyFigure1a(switching_env=self.switching_env,
                                           starting_checkpoint=starting_checkpoint,
                                           checkpoint_duration=checkpoint_duration,
                                           num_checkpoints=num_checkpoints,
                                           num_selected_actions=num_selected_arms)
            iteration_dict['selected_arms'] = our_policy.exploration_actions
            if hasattr(our_policy, 'complexity_value'):
                iteration_dict['complexity_value'] = our_policy.complexity_value
                iteration_dict['min_min_singular_value'] = our_policy.min_min_singular_value
            for n in range(self.num_experiments):
                print("experiment_n: " + str(n))

                our_policy.reset()
                current_state = None

                for i in range(total_horizon+1):
                    if i == 0:
                        current_state = np.random.randint(low=0, high=self.switching_env.num_states)
                    else:
                        current_state = np.random.multinomial(
                                n=1, pvals=transition_matrix[current_state],
                                size=1)[0].argmax()

                    # our policy using explore than commit alg
                    our_policy_chosen_arm = our_policy.choose_arm()
                    reward_dist = state_action_reward_matrix[current_state, our_policy_chosen_arm]
                    reward_index = np.random.multinomial(1, reward_dist, 1)[0].argmax()
                    our_policy.update(our_policy_chosen_arm, reward_index)

                    # states_list[n, i] = current_state

                    if i % 10000 == 0:
                        print(f"{i}-th epsisode")

                estimation_results_list[n] = our_policy.estimation_errors

            run_result[j] = estimation_results_list
            list_of_iteration_dict.append(iteration_dict)


        if self.save_bandit_info:
            f = open(self.bandit_dir_path + '/bandit_info.json', 'w')
            json_file = json.dumps(bandit_info_dict)
            f.write(json_file)
            f.close()

        if self.save_results:
            only_one_result_file = (len(num_selected_arms_list) == 1 and
                    num_selected_arms_list[0] == self.switching_env.num_actions)

            if only_one_result_file:
                # in this case we produce a unique file of results
                result_dict.update(list_of_iteration_dict[0])
                result_dict['estimation_errors'] = run_result[0].tolist()

            f = open(self.exp_dir_path + f'/exp_info.json', 'w')
            json_file = json.dumps(result_dict)
            f.write(json_file)
            f.close()

            if not only_one_result_file:
                for i, elem in enumerate(num_selected_arms_list):
                    f = open(self.exp_dir_path + f'/{elem}_arm.json', 'w')
                    current_dict = list_of_iteration_dict[i]
                    current_dict['result'] = run_result[i].tolist()
                    json_file = json.dumps(current_dict)
                    f.write(json_file)
                    f.close()
