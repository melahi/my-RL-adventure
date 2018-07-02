import gym
import random
import numpy as np


class Memory:
    def __init__(self,
                 observation_space,
                 number_of_actions,
                 needed_observation_for_state=1,
                 memory_capacity=1000000,
                 look_ahead_state_for_reward_estimation=1,
                 sampling_probability=0.3,
                 gamma=0.95):
        assert look_ahead_state_for_reward_estimation > 0, "look_ahead_state_for_reward_estimation should be greater" \
                                                           "than zero."
        assert Memory.__check_environment_assumptions(observation_space), "The given `observation_space` does not" \
                                                                          "satisfy our assumptions"
        self.__last_state = np.zeros(shape=[*observation_space.shape, needed_observation_for_state])
        self.__number_of_actions = number_of_actions
        self.__memory_capacity = memory_capacity
        self.__look_ahead_state_for_reward_estimation = look_ahead_state_for_reward_estimation
        self.__sampling_probability = sampling_probability
        self.__gamma = gamma
        self.__uncompleted_experiences = list()
        self.__memorized_experiences = [None] * self.__memory_capacity
        self.__next_memory_slot = 0

    def reset(self):
        self.__last_state = np.zeros(shape=self.__last_state.shape)
        self.__uncompleted_experiences = list()

    def get_state(self, observation):
        """Completing current state based on new observation."""
        self.__last_state[:, :, :-1] = self.__last_state[:, :, 1:]
        self.__last_state[:, :, -1] = observation
        return self.__last_state

    def save_state(self, state, reward, action):
        if random.random() < self.__sampling_probability:
            self.__uncompleted_experiences.append(Experience(state,
                                                             action,
                                                             self.__look_ahead_state_for_reward_estimation,
                                                             self.__gamma))
        new_uncompleted_experience_list = list()
        for experience in self.__uncompleted_experiences:
            if not experience.add_reward(reward):
                # It means that this experience still needs to know rewards of future states.
                new_uncompleted_experience_list.append(experience)
            else:
                self.__memorized_experiences[self.__next_memory_slot] = experience
                self.__next_memory_slot = (self.__next_memory_slot + 1) % self.__memory_capacity

        self.__uncompleted_experiences = new_uncompleted_experience_list

    def remember_experience(self, batch_size=128):
        memorized_experiences = self.__memorized_experiences
        experience_count = self.__memory_capacity
        if memorized_experiences[self.__next_memory_slot] is None:
            experience_count = self.__next_memory_slot
            memorized_experiences = memorized_experiences[:experience_count]
        random.shuffle(memorized_experiences)
        states_shape = [batch_size, *self.__last_state.shape]
        states = np.zeros(states_shape)
        actions = np.zeros([batch_size], dtype=np.uint8)
        rewards = np.zeros([batch_size, self.__number_of_actions])
        starting_index = 0

        while starting_index < experience_count:
            batch_size = min(starting_index + batch_size, experience_count) - starting_index
            for idx in range(batch_size):
                states[idx] = memorized_experiences[starting_index + idx].state
                actions[idx] = memorized_experiences[starting_index + idx].action
                rewards[idx, actions[idx]] = memorized_experiences[starting_index + idx].reward
            yield (states[:batch_size], {'committed_action': actions[:batch_size], 'q_value': rewards[:batch_size]})
            starting_index += batch_size

    @staticmethod
    def __check_environment_assumptions(observation_space):
        if not hasattr(observation_space, 'shape'):
            print("observation_space does not have 'shape' attribute")
            return False
        if len(observation_space.shape) != 2:
            print("observation_space should be a rank-2 tensor.")
            return False
        return True


class Experience:
    def __init__(self, state, action, look_ahead_state_for_reward_estimation, gamma):
        self.state = state
        self.action = action
        self.reward = 0
        self.__gamma = gamma
        self.__gamma_coefficient = 1
        self.__need_to_know_reward_counter = look_ahead_state_for_reward_estimation
        assert look_ahead_state_for_reward_estimation > 0, "look_ahead_state_for_reward_estimation should be greater" \
                                                           "than zero."

    def add_reward(self, reward):
        """
        Adding reward of next future state to this experience.
        Returns if all rewards of needed look ahead states are added or not.

        :param reward: The reward of next future state
        :return: `True` if all need reward was added, `False` otherwise.
        """
        self.reward += self.__gamma_coefficient * reward
        self.__gamma_coefficient *= self.__gamma
        self.__need_to_know_reward_counter -= 1
        if self.__need_to_know_reward_counter == 0:
            return True
        return False
