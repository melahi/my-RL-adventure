import os
import gym
from argparse import ArgumentParser

from agent import Agent
from memory import Memory
from environment import Environment
from decision_maker import DecisionMaker
from long_term_memory import LongTermMemory


def argument_parsing():
    parser = ArgumentParser()
    parser.add_argument('-e', '--environment', type=str, default='Breakout', help='Name of the game')
    parser.add_argument('-d', '--dir', type=str, default='./rl-adventure/')
    args = parser.parse_args()
    return args.environment, args.dir


def main():
    environment_id, directory = argument_parsing()
    directory = os.path.join(directory, environment_id)
    environment = Environment(environment_id, directory)
    long_term_memory = LongTermMemory(directory)
    memory = long_term_memory.restoring_memory_object()
    if memory is None:
        memory = Memory(environment.observation_space,
                        number_of_actions=environment.action_space.n)
    exploration_rate = long_term_memory.restoring_exploration_object()
    if exploration_rate is None:
        exploration_rate = 1
    decision_maker = DecisionMaker(state_space=environment.observation_space,
                                   number_of_actions=environment.action_space.n,
                                   model_dir=directory,
                                   exploration_rate=exploration_rate)
    agent = Agent(environment, decision_maker, memory)
    agent.play()
    long_term_memory.saving_memory(memory)
    long_term_memory.saving_exploration_rate(decision_maker.get_exploration_rate())


if __name__ == "__main__":
    main()
