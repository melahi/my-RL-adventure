import os
import random
import tensorflow as tf
import multiprocessing

from env.atari_wrappers import LazyFrames


class DecisionMaker:
    PRIMARY_SCOPE = 'decision_making'
    MAIN_NETWORK_NAME = 'main'
    PERSISTED_NETWORK_NAME = 'persisted'

    class PersistingPredictionKnowledgeHook(tf.train.SessionRunHook):
        PERSISTING_OPERATOR_NAME = 'persisting_operator'

        def begin(self):
            self.__create_persisting_prediction_knowledge_network()

        def after_create_session(self, session, coord):
            session.run("{}".format(self.PERSISTING_OPERATOR_NAME))

        def __create_persisting_prediction_knowledge_network(self):
            pattern = '{}/{}'
            source_scope = pattern.format(DecisionMaker.PRIMARY_SCOPE, DecisionMaker.MAIN_NETWORK_NAME)
            destination_scope = pattern.format(DecisionMaker.PRIMARY_SCOPE, DecisionMaker.PERSISTED_NETWORK_NAME)
            source_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, source_scope)
            destination_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, destination_scope)
            assert len(source_variables) == len(destination_variables)
            persisting_ops = []
            for src, dst in zip(source_variables, destination_variables):
                persisting_ops.append(dst.assign(src))
            tf.group(*persisting_ops, name=self.PERSISTING_OPERATOR_NAME)

    def __init__(self,
                 state_space,
                 number_of_actions: int,
                 model_dir: str,
                 learning_rate: float=0.000025,
                 exploration_rate: float=1.0,
                 gamma: float=0.975,
                 training_steps=500):
        self._model_name = "DeepQN"
        self.__learning_rate = learning_rate
        self.__conv_filter_count = [32, 64, 64]
        self.__conv_filter_size = [8, 4, 3]
        self.__conv_stride = [4, 2, 1]
        self.__fully_connected_units = [512]
        self.__number_of_actions = number_of_actions
        self.__state_space = state_space
        self.__decision_process_started = False
        self.__prediction_function = None
        self.__exploration_rate = exploration_rate
        self.__exploration_rate_decay = 0.0001
        self.__gamma = gamma
        self.__training_steps = training_steps
        model_dir = os.path.join(model_dir, "models", self._model_name)
        os.makedirs(model_dir, exist_ok=True)
        self.__model = tf.estimator.Estimator(model_fn=self._model_fn,
                                              model_dir=model_dir)
        # When a decision for a state is needed to be taken, the state will be stored in the `__need_to_predict`
        # variable and then used in the `input_generation_for_prediction` function to provide the input for the model,
        # then the model will make the decision.
        self.__need_to_predict = None
        self.__provided_input = multiprocessing.Semaphore(0)

    def __del__(self):
        self.__terminate_decision_process()

    def making_decision(self, state: LazyFrames=None, is_validation=False):
        """
        Making decision about which action should be performed in the given `state`.
        If `state` be None, it means that decision procedure should be over.

        :param state: The state which the decision should be take for it.
        :param is_validation: should decision be made for validation or not
        :return: The action which should be perform in the given state.
        """
        if state is None:
            self.__terminate_decision_process()
            return
        if not is_validation and random.random() < self.__exploration_rate:
            return random.randrange(self.__number_of_actions)

        if not self.__decision_process_started:
            self.__start_decision_process()
        self.__need_to_predict = [state]
        self.__provided_input.release()
        return next(self.__prediction_function)['selected_action']

    def train(self, training_input_generator, evaluation_input_generator):
        self.__terminate_decision_process()
        features_type, features_shape = self.__get_features_structure()
        labels_type, labels_shape = self.__get_labels_structure()
        types = (features_type, labels_type)
        shapes = (features_shape, labels_shape)
        self.__model.train(input_fn=lambda: tf.data.Dataset.from_generator(training_input_generator,
                                                                           types,
                                                                           shapes),
                           hooks=[self.PersistingPredictionKnowledgeHook()],
                           steps=self.__training_steps)
        eval_result = self.__model.evaluate(input_fn=lambda: tf.data.Dataset.from_generator(evaluation_input_generator,
                                                                                            types,
                                                                                            shapes))
        self.__exploration_rate = max(0.01, self.__exploration_rate - self.__exploration_rate_decay)
        print("Evaluation result:", eval_result)
        print("Exploration rate:", self.__exploration_rate)

    def get_exploration_rate(self):
        return self.__exploration_rate

    def _model_fn(self, features, labels, mode):
        with tf.variable_scope(self.PRIMARY_SCOPE):
            q_value = self.__q_value_network(features, name=self.MAIN_NETWORK_NAME)
            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = {
                    'selected_action': tf.argmax(q_value, axis=1),
                    'estimated_q_value': tf.reduce_max(q_value, axis=1)
                }
                return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
            committed_action = tf.one_hot(indices=labels['committed_action'],
                                          depth=self.__number_of_actions,
                                          dtype=tf.float32)
            q_value = tf.reduce_sum(committed_action * q_value, axis=1)
            next_state_q_value = self.__q_value_network(labels['next_state'], name=self.PERSISTED_NETWORK_NAME)
            next_state_q_value = tf.reduce_max(next_state_q_value, axis=1) * labels['done']
            total_reward = labels['reward'] + (self.__gamma * next_state_q_value)
            total_reward = tf.stop_gradient(total_reward)
            loss = tf.losses.mean_squared_error(labels=total_reward, predictions=q_value)
            tf.summary.scalar('loss', loss)
            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss)

            optimizer = tf.train.AdamOptimizer(learning_rate=self.__learning_rate)
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    def __q_value_network(self, state, name: str):
        with tf.variable_scope(name):
            net = (state - 128) / 256  # Normalizing input
            for layer_index, (filter_count, filter_size, stride) in enumerate(zip(self.__conv_filter_count,
                                                                                  self.__conv_filter_size,
                                                                                  self.__conv_stride)):
                net = tf.layers.conv2d(inputs=net,
                                       filters=filter_count,
                                       kernel_size=filter_size,
                                       strides=stride,
                                       padding='valid',
                                       activation=tf.nn.relu,
                                       name="conv_{}".format(layer_index))
            net = tf.layers.flatten(inputs=net)
            for layer_index, units in enumerate(self.__fully_connected_units):
                net = tf.layers.dense(inputs=net,
                                      units=units,
                                      activation=tf.nn.relu,
                                      name="dens_{}".format(layer_index))
            tf.summary.histogram("LastLayer_" + name, net)
            q_value = tf.layers.dense(inputs=net, units=self.__number_of_actions, activation=None, name="q_value")
            return q_value

    def __input_generator_for_prediction(self):
        while True:
            self.__provided_input.acquire()
            if self.__need_to_predict is None:
                return
            yield self.__need_to_predict

    def __get_features_structure(self):
        features_type = tf.float32
        features_shape = tf.TensorShape([None, *self.__state_space.shape])
        return features_type, features_shape

    def __get_labels_structure(self):
        labels_type = {'next_state': tf.float32, 'reward': tf.float32, 'committed_action': tf.uint8, 'done': tf.float32}
        labels_shape = {'next_state': tf.TensorShape([None, *self.__state_space.shape]),
                        'reward': tf.TensorShape([None]),
                        'committed_action': tf.TensorShape([None]),
                        'done': tf.TensorShape([None])}
        return labels_type, labels_shape

    def __start_decision_process(self):
        if self.__decision_process_started:
            return
        self.__decision_process_started = True
        self.__provided_input = multiprocessing.Semaphore(0)
        features_type, features_shape = self.__get_features_structure()
        self.__prediction_function = self.__model.predict(input_fn=lambda: tf.data.Dataset.from_generator(
            self.__input_generator_for_prediction, features_type, features_shape))

    def __terminate_decision_process(self):
        if not self.__decision_process_started:
            return
        self.__decision_process_started = False
        self.__prediction_function = None
        self.__need_to_predict = None

