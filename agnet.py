class Agent:
    def __init__(self, environment, decision_maker, memory):
        self.__env = environment
        self.__playing = False
        self.__decision_maker = decision_maker
        self.__memory = memory
        self.__episodes_periode = 10

    def play(self):
        self.__playing = True
        episode_counter = 0
        while self.__playing:
            self.__memory.reset()
            observation = self.__env.reset()
            state = self.__memory.get_state(observation)
            episode_finished = False
            while not episode_finished:
                action = self.__decision_maker.making_decision(state)
                self.__env.act(action)
                observation, reward, episode_finished = self.__env.percept()
                state = self.__memory.get_state(observation)
            self.finalizing_episode(episode_counter)
            episode_counter += 1

    def finalizing_episode(self, episode_counter):
        if episode_counter % self.__episodes_periode == 0:
            print("Finishing episode")
        pass
