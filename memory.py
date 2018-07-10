import random
import collections
import numpy as np

from env.atari_wrappers import LazyFrames


class Memory:
    def __init__(self,
                 observation_space,
                 number_of_actions,
                 training_capacity=1000000,
                 validation_capacity=None,
                 sampling_probability=0.5):
        assert Memory.__check_environment_assumptions(observation_space), "The given `observation_space` does not" \
                                                                          "satisfy our assumptions"
        self.__state_shape = observation_space.shape
        self.__number_of_actions = number_of_actions
        self.__training_experiences = collections.deque(maxlen=training_capacity)
        if validation_capacity is None:
            validation_capacity = int(training_capacity / 3)
        self.__validation_experiences = collections.deque(maxlen=validation_capacity)
        self.__validation_sampling_rate = validation_capacity / (training_capacity + validation_capacity)
        self.__sampling_probability = sampling_probability

    def save_state(self, state, reward, action, next_state):
        if random.random() < self.__sampling_probability:
            experience = Experience(state, action, reward, next_state)
            if random.random() < self.__validation_sampling_rate:
                self.__validation_experiences.append(experience)
            else:
                self.__training_experiences.append(experience)

    def remember_training_experiences(self, batch_size=128):
        random.shuffle(self.__training_experiences)
        return self.__remember_experiences(self.__training_experiences, batch_size)

    def remember_evaluation_experiences(self, batch_size=128):
        return self.__remember_experiences(self.__validation_experiences, batch_size)

    def __remember_experiences(self, memorized_experiences, batch_size):
        states_shape = [batch_size, *self.__state_shape]
        states = np.zeros(states_shape)
        actions = np.zeros([batch_size], dtype=np.uint8)
        rewards = np.zeros([batch_size, self.__number_of_actions])
        next_state = np.zeros(states_shape)
        batch_index = 0
        for experience in memorized_experiences:
            states[batch_index] = experience.state
            actions[batch_index] = experience.action
            rewards[batch_index, experience.action] = experience.reward
            next_state[batch_index] = experience.next_state
            batch_index += 1
            if batch_index % batch_size == 0:
                yield (states, {'next_state': next_state, 'committed_action': actions, 'reward': rewards})
                # creating another batch
                batch_index = 0
                states = np.zeros(states_shape)
                actions = np.zeros([batch_size], dtype=np.uint8)
                rewards = np.zeros([batch_size, self.__number_of_actions])
                next_state = np.zeros(states_shape)
        if batch_index != 0:
            yield (states[:batch_index], {'next_state': next_state[:batch_index],
                                          'committed_action': actions[:batch_index],
                                          'reward': rewards[:batch_index]})

    @staticmethod
    def __check_environment_assumptions(observation_space):
        if not hasattr(observation_space, 'shape'):
            print("observation_space does not have 'shape' attribute")
            return False
        if len(observation_space.shape) != 3:
            print("observation_space should be a rank-3 tensor.")
            return False
        return True


class Experience:
    def __init__(self, state: LazyFrames, action: int, reward: float, next_state: LazyFrames):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
