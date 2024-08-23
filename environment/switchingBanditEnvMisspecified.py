import numpy as np
import utils


class SwitchingBanditEnvMisspecified:

    def __init__(self, num_states, num_actions, num_obs,
                 transition_matrix=None,
                 state_action_reward_matrix=None,
                 real_reference_matrix=None,
                 possible_rewards=None,
                 transition_multiplier=0,
                 observation_multiplier=5,
                 min_transition_prob=None,
                 misspecification=0.1
                 ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_obs = num_obs
        self.observation_multiplier = observation_multiplier
        self.transition_multiplier = transition_multiplier
        self.misspecification = misspecification

        if transition_matrix is not None:
            self.transition_matrix = transition_matrix
            self.state_action_reward_matrix = state_action_reward_matrix
            self.possible_rewards = possible_rewards
            self.real_reference_matrix = real_reference_matrix
            # self.misspecified_reference_matrix = misspecified_reference_matrix
            # self.misspecified_state_action_reward_matrix = misspecified_state_action_reward_matrix
        else:
            self.transition_matrix = self.generate_transition_matrix_v2(
                transition_multiplier=transition_multiplier,
                min_transition_prob=min_transition_prob)
            self.state_action_reward_matrix = \
                self.generate_state_action_reward_dist(observation_multiplier=observation_multiplier)
            # with reference matrix we mean the A matrix
            # self.misspecified_state_action_reward_matrix = (
            #     self.generate_misspecified_state_action_reward_dist(
            #         misspecification=misspecification))
            self.real_reference_matrix = self.generate_reference_matrix(
                state_action_reward_matrix=self.state_action_reward_matrix
            )
            # self.misspecified_reference_matrix = self.generate_reference_matrix(
            #     state_action_reward_matrix=self.misspecified_state_action_reward_matrix
            # )
            self.possible_rewards = np.random.permutation(
                np.linspace(start=0.0, stop=1.0, num=self.num_obs))
        self.stationary_distribution = self.compute_stationary_distribution()
        self.transition_stationary_distribution = self.compute_transition_stationary_distribution()

        # self.misspecified_sigma_min = utils.compute_min_svd(self.misspecified_reference_matrix)
        self.real_sigma_min = utils.compute_min_svd(self.real_reference_matrix)
        self.second_eigenvalue = utils.compute_second_eigenvalue(self.transition_matrix)
        print(f"The smaller real sigma is {self.real_sigma_min}")
        # print(f"The smaller misspecified sigma is {self.misspecified_sigma_min}")

    def generate_env(self):
        print(self.transition_matrix)

    def generate_transition_matrix(self, transition_multiplier):
        # by setting specific design we give more probability to self-loops
        diag_matrix = np.eye(self.num_states) * transition_multiplier
        markov_chain = np.random.random_integers(
            low=1, high=11, size=(self.num_states, self.num_states))
        markov_chain = markov_chain + diag_matrix
        transition_matrix = markov_chain / markov_chain.sum(axis=1)[:, None]
        return transition_matrix

    def generate_transition_matrix_v2(self, transition_multiplier, min_transition_prob):
        # by setting specific design we give more probability to self-loops
        diag_matrix = np.eye(self.num_states) * transition_multiplier
        for state in range(self.num_states):
            outcome = np.random.random_integers(low=0, high=self.num_states, size=1)[0]
            while outcome != state:
                outcome = \
                np.random.random_integers(low=0, high=self.num_states, size=1)[
                    0]
            diag_matrix[state, outcome] += transition_multiplier * (2/3)
        markov_chain = np.random.random_integers(
            low=1, high=15, size=(self.num_states, self.num_states))
        markov_chain = markov_chain + diag_matrix
        transition_matrix = markov_chain / markov_chain.sum(axis=1)[:, None]

        if min_transition_prob is not None:
            while np.any(transition_matrix < min_transition_prob):
                transition_matrix[transition_matrix < min_transition_prob] += min_transition_prob / 4
                transition_matrix = (transition_matrix /
                                                transition_matrix.sum(axis=1)[:, None])

        return transition_matrix

    def generate_state_action_reward_dist(self, observation_multiplier):
        state_action_reward_matrix = np.empty(
            shape=(self.num_states, self.num_actions, self.num_obs))
        perturbation_matrix = np.zeros(shape=(self.num_actions, self.num_obs))

        if self.num_actions >= self.num_obs:
            for i in range(int(self.num_actions / self.num_obs)):
                perturbation_matrix[self.num_obs*i:self.num_obs*(i+1), :] = observation_multiplier * np.eye(self.num_obs)
        else:
            for i in range(int(self.num_obs / self.num_actions)):
                perturbation_matrix[:, self.num_actions*i:self.num_actions*(i+1)] = observation_multiplier * np.eye(self.num_actions)

        for state in range(self.num_states):
            action_reward = np.random.random((self.num_actions, self.num_obs))
            if self.num_actions >= self.num_obs:
                permutation = np.random.permutation(self.num_actions)
                permuted_matrix = perturbation_matrix[permutation, :]
            else:
                permutation = np.random.permutation(self.num_obs)
                permuted_matrix = perturbation_matrix[:, permutation]

            action_reward += permuted_matrix
            action_reward = action_reward / action_reward.sum(axis=1)[:, None]
            state_action_reward_matrix[state] = action_reward

            #for action in range(self.num_actions):
            #    categorical = np.random.random(size=self.num_obs)
            #    categorical = categorical / categorical.sum()
            #    state_action_reward_matrix[state, action, :] = \
            #        np.array(categorical)
        return state_action_reward_matrix


    def generate_misspecified_values(self, misspecification):
        self.misspecified_state_action_reward_matrix = self.generate_misspecified_state_action_reward_dist(
            misspecification=misspecification)
        self.misspecified_reference_matrix = self.generate_reference_matrix(
            state_action_reward_matrix=self.misspecified_state_action_reward_matrix
        )

    def generate_misspecified_state_action_reward_dist(self, misspecification):
        misspecified_state_action_reward_matrix = np.empty(
            shape=(self.num_states, self.num_actions, self.num_obs))

        for state in range(self.num_states):
            for action in range(self.num_actions):
                random_perturbation = np.random.rand(self.num_obs) * misspecification
                randomized_obs_dist = self.state_action_reward_matrix[state, action] + random_perturbation
                randomized_obs_dist = randomized_obs_dist / randomized_obs_dist.sum()
                misspecified_state_action_reward_matrix[state, action] = randomized_obs_dist

        return misspecified_state_action_reward_matrix

    def generate_reference_matrix(self, state_action_reward_matrix):
        reference_matrix = np.empty(
            shape=(self.num_actions ** 2 * self.num_obs ** 2,
                   self.num_states ** 2))

        for starting_state in range(self.num_states):
            for arriving_state in range(self.num_states):
                # print(f"From state {starting_state} to {arriving_state}")
                column = starting_state * self.num_states + arriving_state
                for first_action in range(self.num_actions):
                    for second_action in range(self.num_actions):
                        starting_row = first_action * self.num_actions * self.num_obs ** 2 + second_action * self.num_obs ** 2
                        ending_row = starting_row + self.num_obs ** 2
                        first_obs_prob = state_action_reward_matrix[starting_state, first_action]
                        second_obs_prob = state_action_reward_matrix[arriving_state, second_action]
                        obs_probabilities = np.outer(first_obs_prob, second_obs_prob).reshape(-1)
                        reference_matrix[starting_row:ending_row, column] = obs_probabilities
                        # print(obs_probabilities)
        return reference_matrix

    def compute_stationary_distribution(self):
        evals, evecs = np.linalg.eig(self.transition_matrix.T)
        evec1 = evecs[:, np.isclose(evals, 1)]

        evec1 = evec1[:, 0]
        print(f"Stationary distribution before normalization is {evec1}")

        stationary = evec1 / evec1.sum()
        print(f"Stationary distribution is {stationary}")
        print(stationary.real)
        return stationary.real

    def compute_transition_stationary_distribution(self):
        # real_transition_distribution = np.zeros(shape=(self.num_states, self.num_states))
        transition_stationary_distribution = self.stationary_distribution[:, None] * self.transition_matrix
        # for i in range(self.num_states):
        #     real_transition_distribution[i] = self.stationary_distribution[i] * \
        #                                       self.transition_matrix[i]

        print(f"The transition matrix is \n{self.transition_matrix}")
        print(f"The stationary distribution is \n{self.stationary_distribution}")
        print(f"The transition stationary distribution is \n{transition_stationary_distribution}")
        return transition_stationary_distribution





