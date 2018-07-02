import gym
from argparse import ArgumentParser

from agent import Agent
from memory import Memory
from environment import Environment
from decision_maker import DecisionMaker


def argument_parsing():
    parser = ArgumentParser()
    parser.add_argument('-e', '--environment', type=str, default='Breakout', help='Name of the game')
    parser.add_argument('-d', '--dir', type=str, default='./rl-adventure/')
    args = parser.parse_args()
    return args.environment, args.dir


def main():
    environment_id, directory = argument_parsing()
    environment = Environment(environment_id, directory)
    memory = Memory(environment.observation_space, look_ahead_state_for_reward_estimation=10)
    needed_frame_for_state = 1
    state_space = gym.spaces.Box(environment.observation_space.low,
                                 environment.observation_space.high,
                                 shape=[*environment.observation_space.shape, needed_frame_for_state],
                                 dtype=environment.observation_space.dtype)
    decision_maker = DecisionMaker(state_space=state_space, number_of_actions=environment.action_space.n)
    agent = Agent(environment, decision_maker, memory)
    agent.play()


if __name__ == "__main__":
    main()
