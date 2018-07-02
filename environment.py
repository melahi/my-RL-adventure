import gym
from env.atari_wrappers import make_atari, wrap_deepmind


class Environment:
    def __init__(self, monitor_path=None, game='Breakout'):
        self.__create_environment(monitor_path, game)
        self.width_frame = self.__env.width
        self.height_frame = self.__env.height
        self.__last_observation = None
        self.__last_reward = None
        self.__episode_finished = None

    def __create_environment(self, monitor_path, game):
        self.__env = make_atari(game + "NoFrameskip-v4")
        if monitor_path:
            self.__env = gym.wrappers.Monitor(self.__env, directory=monitor_path, resume=True)
        self.__env = wrap_deepmind(self.__env, episode_life=True, clip_rewards=False, frame_stack=False, scale=False)

    def perception(self):
        return self.__last_observation, self.__last_reward, self.__episode_finished

    def action(self, action_id):
        self.__last_observation, self.__last_reward, self.__episode_finished, _ = self.__env.step(action_id)

    def reset(self):
        self.__last_observation = None
        self.__last_reward = None
        self.__episode_finished = None

