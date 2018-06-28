import random

import cv2
import numpy as np


class Memory:
    def __init__(self,
                 observing_width,
                 observing_height,
                 needed_observation_for_state=1,
                 memorizing_experiences_capacity=1000000,
                 look_ahead_state_for_reward_estimation=1,
                 sampling_probability=0.3,
                 gamma=0.95):
        assert look_ahead_state_for_reward_estimation > 0, "look_ahead_state_for_reward_estimation should be greater" \
                                                           "than zero."
        self.__last_state = np.zeros(shape=[observing_width, observing_height, needed_observation_for_state])
        self.__memorizing_experiences_capacity = memorizing_experiences_capacity
        self.__look_ahead_state_for_reward_estimation = look_ahead_state_for_reward_estimation
        self.__sampling_probability = sampling_probability
        self.__gamma = gamma
        self.__uncompleted_experiences = list()
        self.__memorized_experiences = [None] * self.__memorizing_experiences_capacity
        self.__next_memory_slot = 0

    def reset(self):
        self.__last_state = np.zeros(shape=self.__last_state.shape)
        self.__uncompleted_experiences = list()

    def get_state(self, observation):
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
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
        self.__uncompleted_experiences = new_uncompleted_experience_list


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
        if self.need_to_know_reward_counter == 0:
            return True
        return False
