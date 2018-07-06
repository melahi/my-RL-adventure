import os
import pickle


class LongTermMemory:
    def __init__(self, directory: str):
        self.__memory_path = os.path.join(directory, "memory.pkl")
        self.__exploration_rate_path = os.path.join(directory, "exploration_rate.pkl")

    def saving_memory(self, memory):
        self.__saving_object(self.__memory_path, memory)

    def saving_exploration_rate(self, exploration_rate: float):
        self.__saving_object(self.__exploration_rate_path, exploration_rate)

    def restoring_memory_object(self):
        return self.__restoring_object(self.__memory_path)

    def restoring_exploration_object(self):
        return self.__restoring_object(self.__exploration_rate_path)

    @staticmethod
    def __saving_object(path, dumping_object, protocol=pickle.HIGHEST_PROTOCOL):
        with open(path, mode='wb') as serializing_object:
            pickle.dump(dumping_object, serializing_object, protocol)

    @staticmethod
    def __restoring_object(path):
        if os.path.isfile(path):
            with open(path, mode='rb') as serialized_object:
                return pickle.load(serialized_object)
        return None
