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
        self.__training_frequency = 500
        self.__start_to_training = 10000
        self.finalizing_episode(0)
        signal.signal(signal.SIGINT, self.terminate)

    def play(self):
        self.__playing = True
        episode_counter = 1
        while self.__playing:
            observation = self.__env.reset()
            episode_finished = False
            while not episode_finished:
                action_id = self.__decision_maker.making_decision(observation)
                self.__env.action(action_id)
                new_observation, reward, episode_finished = self.__env.perception()
                self.__memory.save_state(observation, reward, action_id, new_observation)
                observation = new_observation
            self.finalizing_episode(episode_counter)
            episode_counter += 1

    def finalizing_episode(self, episode_counter):
        if episode_counter % 100 == 0:
            print("Finishing episode: {}".format(episode_counter))
        if episode_counter < self.__start_to_training:
            return
        if episode_counter % self.__training_frequency == 0:
            print("Start training in episode:", episode_counter)
            self.__decision_maker.train(self.__memory.remember_training_experiences,
                                        self.__memory.remember_evaluation_experiences)

    def terminate(self, signum=signal.SIGINT, frame=None):
        print("Terminating gracefully ...")
        self.__playing = False

