from environment import Environment


class Agent:
    def __init__(self, environment, decision_maker, memory):
        self.__env = environment
        self.__playing = False
        self.__decision_maker = decision_maker
        self.__memory = memory

    def play(self):
        self.__playing = True
        episode_counter = 0
        while self.__playing:
            self.__memory.reset()
            observation = self.__env.reset()
            episode_finished = False
            step_counter = 0
            while not episode_finished:
                state = self.__memory.get_state(observation)
                action = self.__decision_maker.making_decision(state)
                self.__env.act(action)
                new_observation, reward, episode_finished = self.__env.percept()
                new_state = self.__memory.get_state(new_observation)



