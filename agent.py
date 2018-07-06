import signal

from memory import Memory
from decision_maker import DecisionMaker
from environment import Environment


class Agent:
    def __init__(self, environment: Environment, decision_maker: DecisionMaker, memory: Memory):
        self.__env = environment
        self.__playing = False
        self.__decision_maker = decision_maker
        self.__memory = memory
        self.__training_frequency = 50
        self.finalizing_episode(0)
        signal.signal(signal.SIGINT, self.terminate)

    def play(self):
        self.__playing = True
        episode_counter = 1
        while self.__playing:
            self.__memory.reset()
            observation = self.__env.reset()
            state = self.__memory.get_state(observation)
            episode_finished = False
            while not episode_finished:
                action_id = self.__decision_maker.making_decision(state)
                self.__env.action(action_id)
                observation, reward, episode_finished = self.__env.perception()
                self.__memory.save_state(state, reward, action_id)
                state = self.__memory.get_state(observation)
            self.finalizing_episode(episode_counter)
            episode_counter += 1

    def finalizing_episode(self, episode_counter):
        if episode_counter % self.__training_frequency == 0:
            print("Start training in episode:", episode_counter)
            self.__decision_maker.train(self.__memory.remember_training_experiences,
                                        self.__memory.remember_evaluation_experiences)

    def terminate(self, signum=signal.SIGINT, frame=None):
        print("Terminating gracefully ...")
        self.__playing = False

