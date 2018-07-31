import os
import gym

from env.atari_wrappers import make_atari, wrap_deepmind


class Environment:
    """
    A class mostly based on gym's Environment but with different API for supporting async and real-time
    environments.
    """
    def __init__(self, game: str, monitor_path: str=None):
        self.validation_episode = False
        self.__create_environment(monitor_path, game)
        self.observation_space = self.__env.observation_space
        self.action_space = self.__env.action_space
        self.__last_observation = None
        self.__last_reward = None
        self.__episode_finished = None

    def perception(self):
        return self.__last_observation, self.__last_reward, self.__episode_finished

    def action(self, action_id):
        self.__last_observation, self.__last_reward, self.__episode_finished, _ = self.__env.step(action_id)

    def reset(self):
        self.__last_observation = self.__env.reset()
        self.__last_reward = None
        self.__episode_finished = None
        return self.__last_observation

    def __create_environment(self, monitor_path, game):
        self.__env = make_atari(game + "NoFrameskip-v4")
        if monitor_path:
            monitor_path = os.path.join(monitor_path, "monitor")
            self.__env = gym.wrappers.Monitor(self.__env,
                                              directory=monitor_path,
                                              force=True,
                                              video_callable=self.__is_validation)
        self.__env = wrap_deepmind(self.__env, episode_life=True, clip_rewards=False, frame_stack=True, scale=False)

    # noinspection PyUnusedLocal
    def __is_validation(self, episode_index):
        return self.validation_episode

