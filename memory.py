import random
import collections
import numpy as np

from env.atari_wrappers import LazyFrames


class Memory:
    def __init__(self,
                 observation_space,
                 number_of_actions,
                 training_capacity=100000,
                 validation_capacity=None,
                 sampling_probability=1):
        assert Memory.__check_environment_assumptions(observation_space), "The given `observation_space` does not" \
                                                                          "satisfy our assumptions"
        self.__state_shape = observation_space.shape
        self.__number_of_actions = number_of_actions
        self.__training_experiences = collections.deque(maxlen=training_capacity)
        if validation_capacity is None:
            validation_capacity = int(training_capacity / 10)
        self.__validation_experiences = collections.deque(maxlen=validation_capacity)
        self.__validation_sampling_rate = validation_capacity / (training_capacity + validation_capacity) * 0.01
        self.__sampling_probability = sampling_probability
        self.__short_term_memory = collections.deque()
        self.tracking_state = list()
        self.__total_reward = 0

    def __len__(self):
        return len(self.__training_experiences)

    def save_state(self, state, reward, action, next_state, done):
        self.__total_reward += reward
        experience = Experience(state, action, reward, next_state, done)
        self.__short_term_memory.append(experience)
        if done and self.__total_reward > 0:
            self.__total_reward = 0
            if random.random() < self.__validation_sampling_rate:
                while len(self.__short_term_memory) > 0:
                    self.__validation_experiences.append(self.__short_term_memory.pop())
            else:
                if len(self.tracking_state) == 0:
                    for exp in self.__short_term_memory:
                        self.tracking_state.append(exp)
                while len(self.__short_term_memory) > 0:
                    self.__training_experiences.append(self.__short_term_memory.pop())
        elif done:
            self.__short_term_memory.clear()

    def remember_training_experiences(self, batch_size=32):
        random.shuffle(self.__training_experiences)
        return self.__remember_experiences(self.__training_experiences, batch_size)

    def remember_evaluation_experiences(self, batch_size=32):
        return self.__remember_experiences(self.__validation_experiences, batch_size)

    def __remember_experiences(self, memorized_experiences, batch_size):
        states_shape = [batch_size, *self.__state_shape]
        states = np.zeros(states_shape)
        actions = np.zeros([batch_size], dtype=np.uint8)
        rewards = np.zeros([batch_size])
        next_state = np.zeros(states_shape)
        continuing = np.zeros(batch_size)
        batch_index = 0
        for experience in memorized_experiences:
            states[batch_index] = experience.state
            actions[batch_index] = experience.action
            rewards[batch_index] = experience.reward
            next_state[batch_index] = experience.next_state
            continuing[batch_index] = 1.0 - float(experience.done)
            batch_index += 1
            if batch_index % batch_size == 0:
                yield (states, {'next_state': next_state, 'committed_action': actions, 'reward': rewards, 'continuing': continuing})
                # creating another batch
                batch_index = 0
                states = np.zeros(states_shape)
                actions = np.zeros([batch_size], dtype=np.uint8)
                rewards = np.zeros([batch_size])
                next_state = np.zeros(states_shape)
                continuing = np.zeros(batch_size)
        if batch_index != 0:
            yield (states[:batch_index], {'next_state': next_state[:batch_index],
                                          'committed_action': actions[:batch_index],
                                          'reward': rewards[:batch_index],
                                          'continuing': continuing[:batch_index]})

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
    def __init__(self, state: LazyFrames, action: int, reward: float, next_state: LazyFrames, done: bool):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
